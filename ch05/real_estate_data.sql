-- `real_estate` 데이터베이스 스키마 및 테이블 생성 스크립트

-- 1. 기존에 `real_estate` 데이터베이스가 존재하면 삭제합니다.
DROP DATABASE IF EXISTS `real_estate`;

-- 2. 새로운 `real_estate` 데이터베이스를 생성합니다.
CREATE DATABASE `real_estate`;

-- 3. `real_estate`를 현재 사용 데이터베이스로 설정합니다.
USE `real_estate`;

-- 4. `sale_data` 테이블을 생성하고, `ingestion_date` 컬럼의 연도를 기준으로 파티셔닝합니다.
CREATE TABLE `sale_data` (
    `id` INT AUTO_INCREMENT,
    `RCPT_YR` VARCHAR(4) NOT NULL,
    `CGG_CD` VARCHAR(255) NOT NULL,
    `CGG_NM` VARCHAR(255) NOT NULL,
    `STDG_CD` VARCHAR(255) NOT NULL,
    `STDG_NM` VARCHAR(255) NOT NULL,
    `LOTNO_SE` VARCHAR(255) NOT NULL,
    `LOTNO_SE_NM` VARCHAR(255) NOT NULL,
    `MNO` VARCHAR(255) NOT NULL,
    `SNO` VARCHAR(255) NOT NULL,
    `BLDG_NM` VARCHAR(255) NOT NULL,
    `CTRT_DAY` VARCHAR(8) NOT NULL,
    `THING_AMT` FLOAT NOT NULL,
    `ARCH_AREA` FLOAT NOT NULL,
    `LAND_AREA` FLOAT NOT NULL,
    `FLR` FLOAT NOT NULL,
    `RGHT_SE` VARCHAR(255) NOT NULL,
    `RTRCN_DAY` VARCHAR(255) NOT NULL,
    `ARCH_YR` VARCHAR(4) NOT NULL,
    `BLDG_USG` VARCHAR(255) NOT NULL,
    `DCLR_SE` VARCHAR(255) NOT NULL,
    `OPBIZ_RESTAGNT_SGG_NM` VARCHAR(255) NOT NULL,
    `ingestion_date` DATE NOT NULL,
    PRIMARY KEY (`id`, `ingestion_date`)
)
PARTITION BY RANGE (YEAR(`ingestion_date`)) (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION pmax VALUES LESS THAN (MAXVALUE)
);