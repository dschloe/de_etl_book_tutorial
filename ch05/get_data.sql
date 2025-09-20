USE `real_estate`;


SELECT * FROM sale_data PARTITION (p2023);

SELECT * FROM sale_data PARTITION (p2024);

SELECT * FROM sale_data PARTITION (p2025);


