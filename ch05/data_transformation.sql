-- data_transformation.sql

-- 1. 변환된 데이터를 저장할 최종 테이블 생성 (파티셔닝 포함)
CREATE TABLE IF NOT EXISTS daily_weather_summary (
    id INT AUTO_INCREMENT,
    city VARCHAR(255),
    fetch_date DATE,
    temp_celsius DECIMAL(5, 2),
    humidity INT,
    weather_main VARCHAR(255),
    PRIMARY KEY (id, fetch_date)
)
PARTITION BY RANGE (YEAR(fetch_date)) (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);

-- 2. 로우 데이터를 최종 테이블로 변환 및 적재
INSERT INTO daily_weather_summary (city, fetch_date, temp_celsius, humidity, weather_main)
SELECT
    city,
    DATE(fetch_timestamp) AS fetch_date,
    JSON_UNQUOTE(JSON_EXTRACT(raw_json, '$.main.temp')) AS temp_celsius,
    JSON_UNQUOTE(JSON_EXTRACT(raw_json, '$.main.humidity')) AS humidity,
    JSON_UNQUOTE(JSON_EXTRACT(raw_json, '$.weather[0].main')) AS weather_main
FROM
    raw_weather_data
ON DUPLICATE KEY UPDATE
    temp_celsius = VALUES(temp_celsius),
    humidity = VALUES(humidity),
    weather_main = VALUES(weather_main);

-- 3. 로우 데이터 임시 테이블 정리
TRUNCATE TABLE raw_weather_data;