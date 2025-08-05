#!/bin/bash

# --- 설정값 입력 ---
MYSQL_USER="evan"
MYSQL_PASSWORD="123456"  # 여기에 비밀번호 입력 또는 읽도록 설정
MYSQL_HOST="localhost"
MYSQL_PORT="3306"
DB_NAME="classicmodels"
ZIP_URL="https://www.mysqltutorial.org/wp-content/uploads/2023/10/mysqlsampledatabase.zip"
ZIP_FILE="mysqlsampledatabase.zip"
SQL_FILE="mysqlsampledatabase.sql"

# --- 1. 다운로드 ---
echo "Downloading sample database..."
curl -L -o "$ZIP_FILE" "$ZIP_URL"

# --- 2. 압축 해제 ---
echo "Unzipping..."
unzip -o "$ZIP_FILE"

# --- 3. DB 생성 ---
echo "Creating database '$DB_NAME'..."
mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" -h "$MYSQL_HOST" -P "$MYSQL_PORT" -e "DROP DATABASE IF EXISTS $DB_NAME; CREATE DATABASE $DB_NAME;"

# --- 4. 데이터 import ---
echo "Importing data into '$DB_NAME'..."
mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" -h "$MYSQL_HOST" -P "$MYSQL_PORT" "$DB_NAME" < "$SQL_FILE"

# --- 5. 정리 ---
echo "Cleanup..."
rm -f "$ZIP_FILE"

echo "Sample database '$DB_NAME' has been successfully imported!"
