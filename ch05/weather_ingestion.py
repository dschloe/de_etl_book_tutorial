# weather_ingestion.py
import requests
import mysql.connector
import json
from datetime import datetime

# 제공된 API 키 및 URL
API_KEY = "391047b25c8cacd63ff897a734825bac"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# 제공된 DB 설정
DB_CONFIG = {
    'host': 'localhost',
    'user': 'evan',
    'password': '123456',
    'database': 'weather_db'
}

# API 호출을 위한 매개변수
CITY = "Seoul"
params = {
    'q': CITY,
    'appid': API_KEY,
    'units': 'metric'
}

# 1. API 호출
response = requests.get(BASE_URL, params=params)
if response.status_code != 200:
    print(f"Error fetching data: {response.status_code}")
    exit()

raw_data = response.json()

# 2. MySQL에 연결 및 데이터 적재
try:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # raw_weather_data 테이블 생성
    create_table_query = """
    CREATE TABLE IF NOT EXISTS raw_weather_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        city VARCHAR(255),
        fetch_timestamp DATETIME,
        raw_json JSON
    );
    """
    cursor.execute(create_table_query)

    # 데이터 삽입
    insert_query = """
    INSERT INTO raw_weather_data (city, fetch_timestamp, raw_json)
    VALUES (%s, %s, %s)
    """
    data_to_insert = (CITY, datetime.now(), json.dumps(raw_data))
    cursor.execute(insert_query, data_to_insert)

    conn.commit()
    print("Python script: Raw data successfully ingested.")

except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection is closed.")