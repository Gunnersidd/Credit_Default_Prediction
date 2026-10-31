-- DuckDB query templates for the local dashboard's registered `clients` and
-- `monthly_status` tables. Positive codes are grouped only by the observed
-- numeric sign; non-positive values are not assigned undocumented semantics.

-- Portfolio KPIs
SELECT count(*) AS client_count,
       count(DISTINCT ID) AS unique_clients,
       avg("default payment next month") AS historical_next_month_default_share,
       avg(LIMIT_BAL) AS mean_credit_limit,
       median(LIMIT_BAL) AS median_credit_limit
FROM clients;

-- Monthly positive status-code share, with explicit client-month denominator
SELECT month, month_order,
       count(*) AS client_month_records,
       count(*) FILTER (WHERE status_code > 0) AS positive_code_records,
       positive_code_records::DOUBLE / client_month_records AS positive_code_share
FROM monthly_status
GROUP BY month, month_order
ORDER BY month_order;

-- Outcome rates by a precomputed credit-limit band
SELECT LIMIT_BAND AS segment, count(*) AS clients,
       avg("default payment next month") AS historical_next_month_default_share
FROM clients
GROUP BY LIMIT_BAND
ORDER BY LIMIT_BAND;
