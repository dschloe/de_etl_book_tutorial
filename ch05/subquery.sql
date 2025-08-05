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

# 문제 3: 'Sales Rep' 직책을 가진 직원들이 담당하는 모든 고객들의 결제(payments) 내역을 조회하세요.

# 비즈니스 문제: 회사는 일반 영업 직원(Sales Rep)들이 관리하는 고객들의 결제 패턴을 분석하고자 합니다. 이를 통해 영업 성과가 높은 직원들의 공통적인 고객 관리 방식이나 결제 유도 전략을 파악할 수 있습니다. 예를 들어, 특정 직책을 가진 영업 사원이 관리하는 고객들이 어떤 결제 수단을 주로 사용하는지, 혹은 특정 기간에 집중적으로 결제가 이루어지는지 등을 분석할 수 있습니다.

SELECT
    customerNumber,
    checkNumber,
    paymentDate,
    amount
FROM
    payments
WHERE
    customerNumber IN (
        SELECT
            customerNumber
        FROM
            customers
        WHERE
            salesRepEmployeeNumber IN (
                SELECT
                    employeeNumber
                FROM
                    employees
                WHERE
                    jobTitle = 'Sales Rep'
            )
    );

# 문제 4: 2003년 1월에 주문을 한 적이 있는 고객들이 지불한 결제(payments) 내역을 모두 조회하세요.
# 비즈니스 문제: 특정 기간에 주문한 이력이 있는 고객들의 결제 패턴을 분석하여, 주문과 결제 시점 간의 관계를 파악하고자 합니다.
SELECT
    customerNumber,
    checkNumber,
    paymentDate,
    amount
FROM
    payments
WHERE
    customerNumber IN (
        SELECT DISTINCT customerNumber
        FROM orders
        WHERE orderDate BETWEEN '2003-01-01' AND '2003-01-31'
    );

# 문제 5: '1952 Alpine Renault 1300' 제품과 동일한 제품군(productLine)과 제품 공급업체(productVendor)를 가지는 모든 제품을 조회하세요.
# 비즈니스 문제: 특정 인기 제품과 동일한 범주 및 공급망을 공유하는 다른 제품들을 찾아내어, 관련 제품들의 판매 전략을 수립하거나 제품 포트폴리오를 확장하는 데 활용하고자 합니다.
SELECT
    productCode,
    productName,
    productLine,
    productVendor
FROM
    products
WHERE
    (productLine, productVendor) = (
        SELECT
            productLine,
            productVendor
        FROM
            products
        WHERE
            productName = '1952 Alpine Renault 1300'
    );

# 문제 6: 'Steve Patterson'이 속한 사무실과 직책을 동시에 가지고 있는 다른 직원을 조회하세요. 
# 비즈니스 문제: 특정 직원('Steve Patterson')의 근무 환경(사무실)과 역할(직책)이 동일한 동료를 찾아내고자 합니다. 이는 조직 내에서 유사한 업무를 수행하는 인력을 파악하고, 협업 체계를 구축하는 데 활용될 수 있습니다.
SELECT
    employeeNumber,
    firstName,
    lastName,
    officeCode,
    jobTitle
FROM
    employees
WHERE
    (officeCode, jobTitle) = (
        SELECT
            officeCode,
            jobTitle
        FROM
            employees
        WHERE
            firstName = 'Steve' AND lastName = 'Patterson'
    )
    AND NOT (firstName = 'Steve' AND lastName = 'Patterson');