from ucimlrepo import fetch_ucirepo
import duckdb
import pandas as pd

# fetch dataset as Pandas DataFrame
online_retail = fetch_ucirepo(id=352) 
data = online_retail.data.original 

df_model_features = duckdb.sql(
'''
WITH RECURSIVE online_retail AS (
    SELECT
        b.Country                                                   AS country,
        b.Description                                               AS description,
        b.Quantity                                                  AS quantity,
        CAST(STRPTIME(b.InvoiceDate, '%m/%d/%Y %H:%M') AS DATE)     AS invoice_date,
        MIN(invoice_date) OVER(PARTITION BY CustomerID)             AS first_purchase_date,
        b.UnitPrice                                                 AS unit_price,
        CAST(b.CustomerID AS STRING)                                AS customer_id,
        b.InvoiceNo                                                 AS invoice_number,
        b.Quantity * b.UnitPrice                                    AS order_value
    FROM
        data b
    WHERE 
        customer_id IS NOT NULL
),

snapshots AS (
    -- Anchor member - first date in dataset: 1/10/2010
    SELECT
        MIN(invoice_date) AS snapshot,
        MAX(invoice_date) AS max_date
    FROM 
        online_retail

    UNION ALL

    -- Recursive Member
    SELECT
        DATE_ADD(snapshot, INTERVAL 2 MONTH) AS snapshot,
        max_date
    FROM 
        snapshots
    WHERE
        DATE_ADD(snapshot, INTERVAL 2 MONTH) <= max_date
),

customer_snapshots AS (
    SELECT 
        s.snapshot,
        c.customer_id
    FROM 
        snapshots s
    CROSS JOIN 
        (SELECT DISTINCT customer_id FROM online_retail) c
),

cleaned_transactions AS (
    SELECT
        customer_id,
        invoice_number,
        invoice_date,
        description,
        order_value,
        quantity,
        MIN(invoice_date) OVER(PARTITION BY customer_id) AS first_purchase_date,
        CASE WHEN description = 'Discount' THEN 1 ELSE 0 END AS is_discount,
        CASE WHEN description = 'Discount' THEN order_value ELSE 0 END AS discount_value,
        CASE WHEN quantity < 0 THEN 1 ELSE 0 END AS is_return,
        CASE WHEN quantity < 0 THEN order_value * -1 ELSE 0 END AS returns_value,
        CASE WHEN ASCII(description) = ASCII(UPPER(description)) THEN 1 ELSE 0 END AS is_product
    FROM
        online_retail
),

snapshot_behaviour AS (
    SELECT 
        s.snapshot, 
        s.customer_id,
        SUM(c.order_value)                                                      AS snapshot_order_value,
        COUNT(DISTINCT c.invoice_number)                                        AS snapshot_orders_placed,
        SUM(c.order_value) / COUNT(DISTINCT c.invoice_number)                   AS snapshot_avg_order_value,
        COUNT(DISTINCT CASE WHEN c.is_discount = 1 THEN c.invoice_number END)   AS snapshot_total_discounts,
        COUNT(DISTINCT CASE WHEN c.is_return = 1 THEN c.invoice_number END)     AS snapshot_orders_returned,
        COUNT(DISTINCT CASE WHEN c.is_product = 1 THEN c.description END)       AS snapshot_unique_products_ordered,
        SUM(c.discount_value)                                                   AS snapshot_discount_value,
        SUM(c.returns_value)                                                    AS snapshot_returns_value,
        DATE_DIFF('day', MAX(first_purchase_date), MAX(invoice_date))           AS customer_age
    FROM
        customer_snapshots s
    LEFT JOIN
        cleaned_transactions c 
        ON c.customer_id = s.customer_id
        AND c.invoice_date >= s.snapshot  
        AND c.invoice_date < DATE_ADD(s.snapshot, INTERVAL 2 MONTH) -- Looking at only the current snapshot
    GROUP BY
        s.snapshot,
        s.customer_id
),

customer_history AS (
    SELECT
        s.snapshot, 
        s.customer_id,
        SUM(c.order_value)                                                      AS total_order_value,
        COUNT(DISTINCT c.invoice_number)                                        AS total_orders_placed,
        SUM(c.order_value) / COUNT(DISTINCT c.invoice_number)                   AS historic_avg_order_value,
        COUNT(DISTINCT CASE WHEN c.is_discount = 1 THEN c.invoice_number END)   AS total_discounts,
        COUNT(DISTINCT CASE WHEN c.is_return = 1 THEN c.invoice_number END)     AS total_orders_returned,
        COUNT(DISTINCT CASE WHEN c.is_product = 1 THEN c.description END)       AS unique_products_ordered,
        SUM(c.discount_value)                                                   AS total_discount_value,
        SUM(c.returns_value)                                                    AS total_returns_value
    FROM
        customer_snapshots s
    LEFT JOIN
        cleaned_transactions c
        ON c.customer_id = s.customer_id
        AND c.invoice_date < s.snapshot -- Looking at behaviour in all previous snapshots
    GROUP BY
        s.snapshot,
        s.customer_id
)

SELECT
    COALESCE(a.customer_id, b.customer_id) AS customer_id,
    COALESCE(a.snapshot, b.snapshot) AS snapshot,

    a.total_order_value,
    a.total_orders_placed,
    a.historic_avg_order_value,
    a.total_discounts,
    a.total_orders_returned,
    a.unique_products_ordered,
    a.total_discount_value,
    a.total_returns_value,

    b.snapshot_order_value,
    b.snapshot_orders_placed,
    b.snapshot_avg_order_value,
    b.snapshot_total_discounts,
    b.snapshot_orders_returned,
    b.snapshot_unique_products_ordered,
    b.snapshot_discount_value,
    b.snapshot_returns_value,
    b.customer_age,

-- Churn Flag: 1 if the NEXT snapshot period has ZERO or NULL orders
CASE 
    WHEN COALESCE(LEAD(snapshot_orders_placed, 1) OVER (
        PARTITION BY customer_id 
        ORDER BY snapshot
    ), 0) = 0 THEN 1 
    ELSE 0 
END AS customer_churned
FROM
    customer_history a 
LEFT JOIN
    snapshot_behaviour b USING(snapshot, customer_id)

QUALIFY LAG(snapshot_orders_placed, 1, 1) OVER (
    PARTITION BY customer_id
    ORDER BY snapshot
) > 0
'''
).df()

df_model_features = df_model_features.fillna(0)
df_model_features.to_csv('online_retail_cleaned.csv')

print("Successfully created dataset")