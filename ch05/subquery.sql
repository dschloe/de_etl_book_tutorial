# 문제 1: 각 직원의 이름과 함께, 그 직원이 담당하는 고객의 평균 신용 한도를 조회하세요.
# 비즈니스 문제: 영업 사원별로 담당하는 고객들의 평균 신용 한도를 파악하여, 각 직원이 어떤 규모의 고객을 주로 관리하고 있는지 파악하고자 합니다.

SELECT
    e.employeeNumber,
    e.firstName,
    e.lastName,
    (
        SELECT AVG(creditLimit)
        FROM customers AS c
        WHERE c.salesRepEmployeeNumber = e.employeeNumber
    ) AS avg_customer_credit_limit
FROM
    employees AS e;

# 문제 2: 2004년에 주문된 모든 주문의 평균 주문 금액보다 큰 금액의 주문 번호와 총 금액을 조회하세요.
# 비즈니스 문제: 2004년 평균 주문 금액을 초과하는 고액 주문들을 식별하여, 어떤 고객들이 대량 주문을 하는지 분석하고자 합니다.
SELECT
    orderNumber,
    SUM(quantityOrdered * priceEach) AS total_order_amount
FROM
    orderdetails
GROUP BY
    orderNumber
HAVING
    SUM(quantityOrdered * priceEach) > (
        SELECT AVG(total_amount)
        FROM (
            SELECT SUM(quantityOrdered * priceEach) AS total_amount
            FROM orderdetails
            JOIN orders ON orderdetails.orderNumber = orders.orderNumber
            WHERE YEAR(orders.orderDate) = 2004
            GROUP BY orderdetails.orderNumber
        ) AS yearly_orders
    );