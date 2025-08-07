import requests
import mysql.connector
from datetime import datetime
import json
import concurrent.futures

# API 인증 키 및 기본 URL (매매 API)
API_KEY = "4e506f62416a686a3131355164764c6f"
BASE_URL = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/tbLnOpendataRtmsV"

# MySQL 접속 정보
DB_CONFIG = {
    'user': 'evan',
    'password': '123456',
    'host': '127.0.0.1',
    'database': 'real_estate'
}

def get_apartment_sale_data(year):
    """
    특정 연도의 아파트 매매 데이터를 가져옵니다.
    RCPT_YR 파라미터를 사용하여 연도별 데이터를 필터링합니다.
    """
    start_index = 1
    end_index = 100
    url = f"{BASE_URL}/{start_index}/{end_index}/{year}"
    
    print(f"DEBUG: {year}년도 API 요청 URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if not response.text:
            print(f"{year}년도 API 응답이 비어있습니다.")
            return None, year

        data = response.json()
        
        if 'tbLnOpendataRtmsV' in data and 'row' in data['tbLnOpendataRtmsV']:
            sale_list = data['tbLnOpendataRtmsV']['row']
            print(f"DEBUG: {year}년도 API 응답에 {len(sale_list)}개 데이터가 있습니다.")
            
            # 아파트 데이터만 필터링하여 반환
            apartment_data = [item for item in sale_list if "아파트" in item.get('BLDG_USG', '')]
            
            if apartment_data:
                print(f"DEBUG: {year}년도 아파트 데이터 {len(apartment_data)}개를 발견했습니다.")
                return apartment_data, year
            else:
                print(f"{year}년도 API 응답에 아파트 데이터가 없습니다.")
                return None, year
        
        return None, year
        
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생 (연도: {year}): {e}")
        return None, year
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류 (연도: {year}): {e}")
        return None, year
    except Exception as e:
        print(f"알 수 없는 오류 발생 (연도: {year}): {e}")
        return None, year

def insert_sale_data_list(data_list, year):
    """MySQL 데이터베이스에 여러 개의 매매 데이터를 삽입합니다."""
    if not data_list:
        return

    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        insert_count = 0
        
        for data in data_list:
            contract_day_str = data.get('CTRT_DAY')
            
            if not contract_day_str:
                continue

            try:
                ingestion_date = datetime.strptime(str(contract_day_str), '%Y%m%d').date()
            except ValueError:
                print(f"날짜 변환 오류: {contract_day_str} 형식 문제. 건너뜁니다.")
                continue
            
            sql = """
                INSERT INTO sale_data 
                (RCPT_YR, CGG_CD, CGG_NM, STDG_CD, STDG_NM, LOTNO_SE, LOTNO_SE_NM, MNO, SNO, BLDG_NM, CTRT_DAY, THING_AMT, ARCH_AREA, LAND_AREA, FLR, RGHT_SE, RTRCN_DAY, ARCH_YR, BLDG_USG, DCLR_SE, OPBIZ_RESTAGNT_SGG_NM, ingestion_date) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                data.get('RCPT_YR'),
                data.get('CGG_CD'),
                data.get('CGG_NM'),
                data.get('STDG_CD'),
                data.get('STDG_NM'),
                data.get('LOTNO_SE'),
                data.get('LOTNO_SE_NM'),
                data.get('MNO'),
                data.get('SNO'),
                data.get('BLDG_NM'),
                data.get('CTRT_DAY'),
                data.get('THING_AMT'),
                data.get('ARCH_AREA'),
                data.get('LAND_AREA'),
                data.get('FLR'),
                data.get('RGHT_SE'),
                data.get('RTRCN_DAY'),
                data.get('ARCH_YR'),
                data.get('BLDG_USG'),
                data.get('DCLR_SE'),
                data.get('OPBIZ_RESTAGNT_SGG_NM'),
                ingestion_date
            )
            
            cursor.execute(sql, values)
            insert_count += 1
            
        conn.commit()

        print(f"{year}년도 아파트 데이터 {insert_count}개가 성공적으로 삽입되었습니다.")

    except mysql.connector.Error as err:
        print(f"MySQL 오류: {err}")
        conn.rollback()
    except Exception as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
        conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

if __name__ == "__main__":
    years_to_fetch = [2023, 2024, 2025]
    
    print(f"### 부동산 매매 데이터 수집 시작 (연도: {years_to_fetch}) ###")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_year = {executor.submit(get_apartment_sale_data, year): year for year in years_to_fetch}
        
        for future in concurrent.futures.as_completed(future_to_year):
            year = future_to_year[future]
            try:
                data_list, fetched_year = future.result()
                if data_list:
                    insert_sale_data_list(data_list, fetched_year)
                else:
                    print(f"{year}년도 데이터 삽입할 내용이 없습니다.")
            except Exception as exc:
                print(f"{year}년도 데이터 처리 중 예외 발생: {exc}")
    
    print("\n### 모든 매매 데이터 수집 작업이 완료되었습니다. ###")