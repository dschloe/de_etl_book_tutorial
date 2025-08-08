-- create_datamart.sql

-- 현재 사용할 데이터베이스를 설정합니다.
USE `real_estate`;

-- 기존에 'dm_sale_summary' 테이블이 존재하면 삭제하여 스크립트 재실행을 용이하게 합니다.
DROP TABLE IF EXISTS `dm_sale_summary`;

-- 1. 분석용 데이터 마트 테이블을 생성합니다.
CREATE TABLE `dm_sale_summary` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `cgg_name` VARCHAR(255) NOT NULL,
    `sale_year` INT NOT NULL,
    `sale_month` INT NOT NULL,
    `avg_price_krw` BIGINT NOT NULL
);

-- 2. 'sale_data' 테이블의 'p2023' 파티션 데이터를 가공하여 삽입합니다.
INSERT INTO `dm_sale_summary` (`cgg_name`, `sale_year`, `sale_month`, `avg_price_krw`)
SELECT
    CGG_NM AS cgg_name,
    YEAR(CTRT_DAY) AS sale_year,
    MONTH(CTRT_DAY) AS sale_month,
    CAST(AVG(THING_AMT) AS SIGNED) AS avg_price_krw
FROM
    sale_data PARTITION (p2023)
GROUP BY
    CGG_NM, YEAR(CTRT_DAY), MONTH(CTRT_DAY);

-- 3. 'sale_data' 테이블의 'p2024' 파티션 데이터를 가공하여 삽입합니다.
INSERT INTO `dm_sale_summary` (`cgg_name`, `sale_year`, `sale_month`, `avg_price_krw`)
SELECT
    CGG_NM AS cgg_name,
    YEAR(CTRT_DAY) AS sale_year,
    MONTH(CTRT_DAY) AS sale_month,
    CAST(AVG(THING_AMT) AS SIGNED) AS avg_price_krw
FROM
    sale_data PARTITION (p2024)
GROUP BY
    CGG_NM, YEAR(CTRT_DAY), MONTH(CTRT_DAY);

-- 4. 'sale_data' 테이블의 'p2025' 파티션 데이터를 가공하여 삽입합니다.
INSERT INTO `dm_sale_summary` (`cgg_name`, `sale_year`, `sale_month`, `avg_price_krw`)
SELECT
    CGG_NM AS cgg_name,
    YEAR(CTRT_DAY) AS sale_year,
    MONTH(CTRT_DAY) AS sale_month,
    CAST(AVG(THING_AMT) AS SIGNED) AS avg_price_krw
FROM
    sale_data PARTITION (p2025)
GROUP BY
    CGG_NM, YEAR(CTRT_DAY), MONTH(CTRT_DAY);

-- 5. 데이터 마트가 올바르게 생성되었는지 확인하기 위해 상위 10개 행을 조회합니다.
SELECT * FROM `dm_sale_summary` LIMIT 10;

-- 6. '강남구'의 2024년 월별 평균 거래 금액을 조회하여 특정 데이터가 올바른지 확인합니다.
SELECT
    cgg_name,
    sale_year,
    sale_month,
    avg_price_krw AS average_price
FROM
    dm_sale_summary
WHERE
    cgg_name = '관악구' AND sale_year = 2025
ORDER BY
    sale_month;