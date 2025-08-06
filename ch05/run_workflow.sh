#!/bin/bash

# --- 스크립트 설정 ---
PYTHON_SCRIPT="weather_ingestion.py"
SQL_SCRIPT="data_transformation.sql"
MYSQL_USER="evan"
MYSQL_PASSWORD="123456"
MYSQL_DB="weather_db"

echo "--- [1/2] Python 스크립트를 실행하여 데이터 수집 시작 ---"
# python3 명령어로 Python 스크립트 실행
python3 "$PYTHON_SCRIPT"

if [ $? -ne 0 ]; then
    echo "오류: Python 스크립트 실행 실패."
    exit 1
fi

echo "--- [2/2] SQL 스크립트를 실행하여 데이터 변환 시작 ---"
# mysql 명령어로 SQL 파일 실행
mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DB" --batch < "$SQL_SCRIPT"

if [ $? -ne 0 ]; then
    echo "오류: SQL 스크립트 실행 실패."
    exit 1
fi

echo "--- 워크플로우가 성공적으로 완료되었습니다. ---"