-- Staging -> finals: cast non-TEXT columns inside a subquery, filter NOT NULL columns outside.
-- Date check: starts with YYYY-MM-DD (handles timestamps, timezones, and date-only values).
-- No deduplication.

TRUNCATE TABLE finals_booking_rooms;
TRUNCATE TABLE rejected_booking_rooms;

INSERT INTO finals_booking_rooms (
    resid,
    ref,
    book_owner,
    "booked on",
    property_name,
    arrival_date,
    departure_date,
    nights,
    custid,
    customer_notes,
    cust_country,
    date_cancelled,
    status,
    pax,
    unit_code,
    room_code,
    room_amount,
    extras_amount,
    tot_amount,
    pay_amount,
    madeby,
    voucher,
    balance,
    is_frozen
)
SELECT
    processed.resid,
    processed.ref,
    processed.book_owner,
    processed."booked on",
    processed.property_name,
    processed.arrival_date,
    processed.departure_date,
    processed.nights,
    processed.custid,
    processed.customer_notes,
    processed.cust_country,
    processed.date_cancelled,
    processed.status,
    processed.pax,
    processed.unit_code,
    processed.room_code,
    processed.room_amount,
    processed.extras_amount,
    processed.tot_amount,
    processed.pay_amount,
    processed.madeby,
    processed.voucher,
    processed.balance,
    (
        COALESCE(processed.departure_date::DATE < CURRENT_DATE, FALSE)
        OR processed.date_cancelled IS NOT NULL
    ) AS is_frozen
FROM (
    SELECT
        resid,
        ref,
        book_owner,
        COALESCE(property_name, '136 on Bealey') AS property_name,
        custid,
        customer_notes,
        cust_country,
        unit_code,
        room_code,
        madeby,
        voucher,

        CASE
            WHEN "booked on" IN ('', 'nan') THEN NULL
            WHEN "booked on" ~ '^\d{4}-\d{2}-\d{2}' THEN "booked on"::TIMESTAMP
            ELSE NULL
        END AS "booked on",

        CASE
            WHEN arrival_date IN ('', 'nan') THEN NULL
            WHEN arrival_date ~ '^\d{4}-\d{2}-\d{2}' THEN arrival_date::TIMESTAMP
            ELSE NULL
        END AS arrival_date,

        CASE
            WHEN departure_date IN ('', 'nan') THEN NULL
            WHEN departure_date ~ '^\d{4}-\d{2}-\d{2}' THEN departure_date::TIMESTAMP
            ELSE NULL
        END AS departure_date,

        CASE
            WHEN date_cancelled IN ('', 'nan') THEN NULL
            WHEN date_cancelled ~ '^\d{4}-\d{2}-\d{2}' THEN date_cancelled::TIMESTAMP
            ELSE NULL
        END AS date_cancelled,

        CASE
            WHEN nights IN ('', 'nan') THEN NULL
            WHEN nights ~ '^-?\d+$' THEN nights::INTEGER
            ELSE NULL
        END AS nights,

        CASE
            WHEN status IN ('', 'nan') THEN NULL
            WHEN status ~ '^-?\d+$' THEN status::INTEGER
            ELSE NULL
        END AS status,

        CASE
            WHEN pax IN ('', 'nan') THEN NULL
            WHEN pax ~ '^-?\d+$' THEN pax::INTEGER
            ELSE NULL
        END AS pax,

        CASE
            WHEN room_amount IN ('', 'nan') THEN NULL
            WHEN room_amount ~ '^-?\d+(\.\d+)?$' THEN room_amount::DOUBLE PRECISION
            ELSE NULL
        END AS room_amount,

        CASE
            WHEN extras_amount IN ('', 'nan') THEN NULL
            WHEN extras_amount ~ '^-?\d+(\.\d+)?$' THEN extras_amount::DOUBLE PRECISION
            ELSE NULL
        END AS extras_amount,

        CASE
            WHEN tot_amount IN ('', 'nan') THEN NULL
            WHEN tot_amount ~ '^-?\d+(\.\d+)?$' THEN tot_amount::DOUBLE PRECISION
            ELSE NULL
        END AS tot_amount,

        CASE
            WHEN pay_amount IN ('', 'nan') THEN NULL
            WHEN pay_amount ~ '^-?\d+(\.\d+)?$' THEN pay_amount::DOUBLE PRECISION
            ELSE NULL
        END AS pay_amount,

        CASE
            WHEN balance IN ('', 'nan') THEN NULL
            WHEN balance ~ '^-?\d+(\.\d+)?$' THEN balance::DOUBLE PRECISION
            ELSE NULL
        END AS balance

    FROM staging_booking_rooms
) AS processed
WHERE processed."booked on" IS NOT NULL
AND processed.ref IS NOT NULL
AND processed.arrival_date IS NOT NULL
AND processed.departure_date IS NOT NULL;

INSERT INTO rejected_booking_rooms (
    resid,
    ref,
    book_owner,
    "booked on",
    property_name,
    arrival_date,
    departure_date,
    nights,
    custid,
    customer_notes,
    cust_country,
    date_cancelled,
    status,
    pax,
    unit_code,
    room_code,
    room_amount,
    extras_amount,
    tot_amount,
    pay_amount,
    madeby,
    voucher,
    balance,
    is_frozen
)
SELECT
    s.resid,
    s.ref,
    s.book_owner,
    s."booked on",
    s.property_name,
    s.arrival_date,
    s.departure_date,
    s.nights,
    s.custid,
    s.customer_notes,
    s.cust_country,
    s.date_cancelled,
    s.status,
    s.pax,
    s.unit_code,
    s.room_code,
    s.room_amount,
    s.extras_amount,
    s.tot_amount,
    s.pay_amount,
    s.madeby,
    s.voucher,
    s.balance,
    NULL
FROM staging_booking_rooms s
WHERE CASE
        WHEN s."booked on" IN ('', 'nan') THEN NULL
        WHEN s."booked on" ~ '^\d{4}-\d{2}-\d{2}' THEN s."booked on"::TIMESTAMP
        ELSE NULL
    END IS NULL;

