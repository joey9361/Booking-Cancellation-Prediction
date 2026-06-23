-- Merge validated finals_booking_rooms into serving_booking_rooms.
-- Run after validation.sql. One import batch at a time.
--
-- A) Frozen in serving -> skip
-- B) New booking -> insert
-- C) Active in serving, active in final -> delete + insert
-- D) Active in serving, frozen in final -> coalesce + delete + insert

-- One row per booking in this batch, with merge action.
CREATE TEMP TABLE merge_plan ON COMMIT DROP AS
SELECT
    f.ref,
    f.property_name,
    f.arrival_date,
    f.departure_date,
    f.is_frozen AS final_frozen,
    s.serving_frozen,
    CASE
        WHEN s.ref IS NULL THEN 'B' -- completely new booking
        WHEN s.serving_frozen THEN 'A' -- frozen in serving meaning it should be skipped
        WHEN NOT f.is_frozen THEN 'C' -- active in serving, active in final
        ELSE 'D' -- active in serving, frozen in final
    END AS action
FROM (
    SELECT DISTINCT
        ref,
        property_name,
        arrival_date,
        departure_date,
        is_frozen
    FROM finals_booking_rooms
) f
LEFT JOIN (
    SELECT
        ref,
        property_name,
        arrival_date,
        departure_date,
        BOOL_OR(is_frozen) AS serving_frozen
    FROM serving_booking_rooms
    GROUP BY ref, property_name, arrival_date, departure_date
) s
    ON f.ref = s.ref
   AND f.property_name = s.property_name
   AND f.arrival_date = s.arrival_date
   AND f.departure_date = s.departure_date;


-- B) New bookings: insert all final room rows.
INSERT INTO serving_booking_rooms (
    resid, ref, book_owner, "booked on", property_name,
    arrival_date, departure_date, nights, custid, customer_notes,
    cust_country, date_cancelled, status, pax, unit_code, room_code,
    room_amount, extras_amount, tot_amount, pay_amount, madeby, voucher,
    balance, is_frozen
)
SELECT
    f.resid, f.ref, f.book_owner, f."booked on", f.property_name,
    f.arrival_date, f.departure_date, f.nights, f.custid, f.customer_notes,
    f.cust_country, f.date_cancelled, f.status, f.pax, f.unit_code, f.room_code,
    f.room_amount, f.extras_amount, f.tot_amount, f.pay_amount, f.madeby, f.voucher,
    f.balance, f.is_frozen
FROM finals_booking_rooms f
INNER JOIN merge_plan m
    ON f.ref = m.ref
   AND f.property_name = m.property_name
   AND f.arrival_date = m.arrival_date
   AND f.departure_date = m.departure_date
WHERE m.action = 'B';


-- D) Build coalesced rows before delete (active -> frozen / cancelled).
CREATE TEMP TABLE transition_rows ON COMMIT DROP AS
SELECT
    n.resid,
    n.ref,
    n.book_owner,
    n."booked on",
    n.property_name,
    n.arrival_date,
    n.departure_date,
    n.nights,
    n.custid,
    n.customer_notes,
    n.cust_country,
    n.date_cancelled,
    n.status,
    n.pax,
    COALESCE(NULLIF(n.unit_code, ''), o.unit_code) AS unit_code,
    CASE
        WHEN n.room_code IN ('dummy', '136') OR n.room_code IS NULL
            THEN COALESCE(o.room_code, n.room_code)
        ELSE n.room_code
    END AS room_code,
    CASE
        WHEN COALESCE(n.room_amount, 0) = 0 AND COALESCE(o.room_amount, 0) > 0
            THEN o.room_amount
        ELSE n.room_amount
    END AS room_amount,
    CASE
        WHEN COALESCE(n.extras_amount, 0) = 0 AND COALESCE(o.extras_amount, 0) > 0
            THEN o.extras_amount
        ELSE n.extras_amount
    END AS extras_amount,
    CASE
        WHEN COALESCE(n.tot_amount, 0) = 0 AND COALESCE(o.tot_amount, 0) > 0
            THEN o.tot_amount
        ELSE n.tot_amount
    END AS tot_amount,
    CASE
        WHEN COALESCE(n.pay_amount, 0) = 0 AND COALESCE(o.pay_amount, 0) > 0
            THEN o.pay_amount
        ELSE n.pay_amount
    END AS pay_amount,
    n.madeby,
    n.voucher,
    CASE
        WHEN COALESCE(n.balance, 0) = 0 AND COALESCE(o.balance, 0) > 0
            THEN o.balance
        ELSE n.balance
    END AS balance,
    n.is_frozen
FROM finals_booking_rooms n
INNER JOIN merge_plan m
    ON n.ref = m.ref
   AND n.property_name = m.property_name
   AND n.arrival_date = m.arrival_date
   AND n.departure_date = m.departure_date
LEFT JOIN LATERAL (
    SELECT o.*
    FROM serving_booking_rooms o
    WHERE o.ref = n.ref
      AND o.property_name = n.property_name
      AND o.arrival_date = n.arrival_date
      AND o.departure_date = n.departure_date
      AND n.room_code IS NOT NULL
      AND o.room_code = n.room_code
    ORDER BY o.room_amount DESC NULLS LAST
    LIMIT 1
) o ON TRUE
WHERE m.action = 'D';


-- C + D) Remove existing serving rows for bookings being replaced.
DELETE FROM serving_booking_rooms s
USING merge_plan m
WHERE s.ref = m.ref
  AND s.property_name = m.property_name
  AND s.arrival_date = m.arrival_date
  AND s.departure_date = m.departure_date
  AND m.action IN ('C', 'D');


-- C) Active refresh: insert current final rows.
INSERT INTO serving_booking_rooms (
    resid, ref, book_owner, "booked on", property_name,
    arrival_date, departure_date, nights, custid, customer_notes,
    cust_country, date_cancelled, status, pax, unit_code, room_code,
    room_amount, extras_amount, tot_amount, pay_amount, madeby, voucher,
    balance, is_frozen
)
SELECT
    f.resid, f.ref, f.book_owner, f."booked on", f.property_name,
    f.arrival_date, f.departure_date, f.nights, f.custid, f.customer_notes,
    f.cust_country, f.date_cancelled, f.status, f.pax, f.unit_code, f.room_code,
    f.room_amount, f.extras_amount, f.tot_amount, f.pay_amount, f.madeby, f.voucher,
    f.balance, f.is_frozen
FROM finals_booking_rooms f
INNER JOIN merge_plan m
    ON f.ref = m.ref
   AND f.property_name = m.property_name
   AND f.arrival_date = m.arrival_date
   AND f.departure_date = m.departure_date
WHERE m.action = 'C';


-- D) Cancelled / checked-out transition: insert coalesced rows.
INSERT INTO serving_booking_rooms (
    resid, ref, book_owner, "booked on", property_name,
    arrival_date, departure_date, nights, custid, customer_notes,
    cust_country, date_cancelled, status, pax, unit_code, room_code,
    room_amount, extras_amount, tot_amount, pay_amount, madeby, voucher,
    balance, is_frozen
)
SELECT
    resid, ref, book_owner, "booked on", property_name,
    arrival_date, departure_date, nights, custid, customer_notes,
    cust_country, date_cancelled, status, pax, unit_code, room_code,
    room_amount, extras_amount, tot_amount, pay_amount, madeby, voucher,
    balance, is_frozen
FROM transition_rows;

-- A) Frozen bookings in serving: no action.

