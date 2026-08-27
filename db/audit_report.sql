-- Audit report: per-user daily query volume plus a trailing 7-day count,
-- so a reviewer can spot a user whose query volume is spiking without
-- scanning the raw query_log row by row.
--
-- CTE: daily_counts pre-aggregates query_log into one row per (user, day).
-- Window function: SUM(...) OVER (PARTITION BY username ORDER BY query_date
-- ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) computes a rolling 7-day total
-- per user without a self-join.

WITH daily_counts AS (
    SELECT
        u.username,
        r.name AS role_name,
        date_trunc('day', ql.queried_at) AS query_date,
        count(*) AS queries_that_day
    FROM query_log ql
    JOIN users u ON u.id = ql.user_id
    JOIN roles r ON r.id = u.role_id
    GROUP BY u.username, r.name, date_trunc('day', ql.queried_at)
)
SELECT
    username,
    role_name,
    query_date,
    queries_that_day,
    sum(queries_that_day) OVER (
        PARTITION BY username
        ORDER BY query_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS trailing_7day_queries
FROM daily_counts
ORDER BY username, query_date;
