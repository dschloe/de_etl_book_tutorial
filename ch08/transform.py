# transform.py

import duckdb
import os
from pathlib import Path

# JSON 파일 경로 설정 (기존 스크래핑 결과)
SOURCE_JSON_PATH = Path("data/books_5pages.json")
# DuckDB 데이터베이스 파일 경로 (영구 저장)
DB_PATH = "data/books_data.duckdb"

def transform_data():
    """
    JSON 데이터를 DuckDB로 로드하고 변환하여 새 테이블에 저장합니다.
    """
    if not os.path.exists(SOURCE_JSON_PATH):
        print(f"오류: 원본 JSON 파일 '{SOURCE_JSON_PATH}'을 찾을 수 없습니다.")
        return

    print(f"DuckDB 데이터베이스 '{DB_PATH}'에 연결 중...")
    con = duckdb.connect(database=DB_PATH)

    try:
        # 1. JSON 파일에서 데이터 로드
        print("JSON 데이터를 임시 테이블에 로드 중...")
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW books_raw AS
            SELECT * FROM read_json_auto('{SOURCE_JSON_PATH}');
        """)

        # 2. 데이터 변환 및 정리
        # 가격을 문자열에서 float 타입으로 변환하고, image_url을 절대 경로로 변환합니다.
        print("데이터를 변환하여 'books_clean' 테이블에 저장 중...")
        con.execute("""
            CREATE OR REPLACE TABLE books_clean AS
            SELECT
                upc,
                title,
                product_type,
                -- 가격 문자열에서 숫자만 추출하여 FLOAT 타입으로 변환
                CAST(regexp_extract(price_excl_tax, '[0-9\\.]+') AS FLOAT) AS price_excl_tax,
                CAST(regexp_extract(price_incl_tax, '[0-9\\.]+') AS FLOAT) AS price_incl_tax,
                CAST(regexp_extract(tax, '[0-9\\.]+') AS FLOAT) AS tax,
                availability,
                -- 'Number of reviews'를 INTEGER 타입으로 변환
                TRY_CAST(num_reviews AS INTEGER) AS num_reviews,
                description,
                image_url,
                url,
                image_path
            FROM books_raw;
        """)

        print("\n변환된 데이터 'books_clean' 테이블 생성 완료.")
        
        # 변환 결과 확인 (옵션)
        print("\n--- 'books_clean' 테이블의 샘플 데이터 ---")
        sample_query = con.execute("SELECT title, price_incl_tax, num_reviews, image_path FROM books_clean LIMIT 5").fetchall()
        for row in sample_query:
            print(f"제목: {row[0]}, 가격: {row[1]}, 리뷰 수: {row[2]}, 이미지 경로: {row[3]}")

    except duckdb.Error as e:
        print(f"DuckDB 오류가 발생했습니다: {e}")
    finally:
        con.close()
        print("\nDuckDB 연결이 종료되었습니다.")

if __name__ == "__main__":
    transform_data()