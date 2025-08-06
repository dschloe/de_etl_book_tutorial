# -- Active: 1754285776339@@127.0.0.1@3306@classicmodels
# 문제 1: 각 사무실별 매출액 상위 1등 직원 조회하세요
# 비즈니스 문제: 영업 관리팀은 각 사무실(officeCode)에서 가장 높은 매출을 올리는 직원 1명을 파악하여, 우수 영업 사원에 대한 보상 계획을 수립하고 모범 사례를 공유하고자 한다. 이 분석을 통해 직원들의 성과를 공정하게 평가하고 동기 부여를 강화할 수 있다. 

SELECT
    officeCode,
    employeeName,
    totalSales,
    ranking
FROM (
    SELECT
        e.officeCode,
        CONCAT(e.firstName, ' ', e.lastName) AS employeeName,
        SUM(p.amount) AS totalSales,
        RANK() OVER (PARTITION BY e.officeCode ORDER BY SUM(p.amount) DESC) AS ranking
    FROM
        employees AS e
    JOIN
        customers AS c ON e.employeeNumber = c.salesRepEmployeeNumber
    JOIN
        payments AS p ON c.customerNumber = p.customerNumber
    GROUP BY
        e.officeCode, employeeName
) AS sales_ranking
WHERE ranking <= 1
ORDER BY officeCode, ranking;

# 문제 2: 각 고객별 가장 최근 3건의 주문 정보 조회하세요
# 비즈니스 문제: 고객 관리팀은 각 고객의 최근 구매 패턴을 분석하여, 재구매 주기를 파악하고 개인화된 마케팅 캠페인을 기획하고자 합니다. 이를 통해 고객 충성도를 높이고, 이탈 가능성이 있는 고객을 조기에 식별할 수 있습니다.

SELECT
    customerNumber,
    orderNumber,
    orderDate,
    recent_order_rank,
    productName,
    quantityOrdered
FROM (
    SELECT
        o.customerNumber,
        o.orderNumber,
        o.orderDate,
        od.productCode,
        od.quantityOrdered,
        ROW_NUMBER() OVER (PARTITION BY o.customerNumber ORDER BY o.orderDate DESC) AS recent_order_rank
    FROM
        orders AS o
    JOIN
        orderdetails AS od ON o.orderNumber = od.orderNumber
) AS ranked_orders
JOIN
    products AS p ON ranked_orders.productCode = p.productCode
WHERE
    recent_order_rank <= 3
ORDER BY
    customerNumber, orderDate DESC, productName;

# 문제 3. 연도별 월별 매출액과 누적 매출액 조회
# 비즈니스 문제: 재무 분석팀은 매월의 매출액과 함께 해당 연도의 누적 매출액을 확인하여, 월별 실적이 연간 목표 달성에 얼마나 기여했는지 파악하고자 합니다. 이를 통해 특정 기간의 실적 부진 원인을 분석하거나, 남은 기간의 목표 달성 전략을 수립할 수 있습니다.
SELECT
    sale_year,
    sale_month,
    monthly_sales,
    SUM(monthly_sales) OVER (PARTITION BY sale_year ORDER BY sale_month) AS cumulative_sales
FROM (
    SELECT
        YEAR(o.orderDate) AS sale_year,
        MONTH(o.orderDate) AS sale_month,
        SUM(od.quantityOrdered * od.priceEach) AS monthly_sales
    FROM
        orders AS o
    JOIN
        orderdetails AS od ON o.orderNumber = od.orderNumber
    GROUP BY
        sale_year, sale_month
) AS monthly_sales_summary
ORDER BY sale_year, sale_month;

# 문제 4. 각 고객의 결제 금액과 해당 고객의 평균 결제 금액 조회하세요
# 비즈니스 문제 : 고객 관계 관리(CRM)팀은 각 고객의 개별 결제 건과 그 고객의 전체 평균 결제액을 비교하여, 특정 결제액이 고객의 일반적인 구매 습관과 얼마나 다른지 파악하고자 합니다. 이를 통해 예상치 못한 고액 결제에 대해 특별한 관리를 하거나, 저액 결제가 반복되는 고객에게 맞춤형 상품을 추천할 수 있습니다.

SELECT
    customerNumber,
    checkNumber,
    paymentDate,
    amount,
    AVG(amount) OVER (PARTITION BY customerNumber) AS avg_customer_payment
FROM
    payments
ORDER BY customerNumber, paymentDate;

# 문제 5: 고객의 연속된 주문 간의 금액 차이와 기간(일수) 차이 분석하세요. 
# 비즈니스 문제 : 영업 전략팀은 고객이 연속적으로 구매한 주문 금액의 변화와 함께, 재구매까지 걸린 시간(일수)을 분석하고자 합니다. 이 데이터를 통해 특정 프로모션이 재구매 주기를 단축시키는지, 또는 가격 변동이 구매 결정에 어떤 영향을 미치는지 등을 파악하여 더욱 정교한 마케팅 전략을 수립할 수 있습니다.
SELECT
    customerNumber,
    orderDate,
    total_amount,
    LAG(total_amount, 1, 0) OVER (PARTITION BY customerNumber ORDER BY orderDate) AS previous_order_amount,
    (total_amount - LAG(total_amount, 1, 0) OVER (PARTITION BY customerNumber ORDER BY orderDate)) AS amount_difference,
    DATEDIFF(orderDate, LAG(orderDate, 1, orderDate) OVER (PARTITION BY customerNumber ORDER BY orderDate)) AS days_since_last_order
FROM (
    SELECT
        o.customerNumber,
        o.orderDate,
        SUM(od.quantityOrdered * od.priceEach) AS total_amount
    FROM
        orders AS o
    JOIN
        orderdetails AS od ON o.orderNumber = od.orderNumber
    GROUP BY
        o.customerNumber, o.orderDate
) AS customer_orders
ORDER BY customerNumber, orderDate;

# 문제 6. 3개월 이동 평균 매출 계산 (ROWS)하세요. 
# 비즈니스 문제 : 영업 분석가는 월별 매출액의 변동성을 줄이고 장기적인 추세를 파악하기 위해, 매월의 매출을 해당 월을 포함한 이전 2개월의 평균으로 계산하고자 합니다. 이를 3개월 이동 평균이라고 합니다. 이 분석을 통해 계절적 요인이나 일시적인 이벤트로 인한 매출 급등락을 보정하고, 보다 안정적인 매출 추세를 시각적으로 확인할 수 있습니다.

SELECT
    sale_year,
    sale_month,
    monthly_sales,
    AVG(monthly_sales) OVER (
        ORDER BY sale_year, sale_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3_months
FROM (
    SELECT
        YEAR(o.orderDate) AS sale_year,
        MONTH(o.orderDate) AS sale_month,
        SUM(od.quantityOrdered * od.priceEach) AS monthly_sales
    FROM
        orders AS o
    JOIN
        orderdetails AS od ON o.orderNumber = od.orderNumber
    GROUP BY
        sale_year, sale_month
) AS monthly_sales_data
ORDER BY sale_year, sale_month;