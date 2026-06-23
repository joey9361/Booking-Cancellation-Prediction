CREATE TABLE IF NOT EXISTS serving_booking_rooms (
    row_id BIGSERIAL PRIMARY KEY,
    resid TEXT,
    ref TEXT NOT NULL,
    book_owner TEXT,
    "booked on" TIMESTAMP NOT NULL,
    property_name TEXT,
    arrival_date TIMESTAMP NOT NULL,
    departure_date TIMESTAMP NOT NULL,
    nights INTEGER,
    custid TEXT,
    customer_notes TEXT,
    cust_country TEXT,
    date_cancelled TIMESTAMP,
    "status" INTEGER,
    pax INTEGER,
    unit_code TEXT,
    room_code TEXT,
    room_amount DOUBLE PRECISION,
    extras_amount DOUBLE PRECISION,
    tot_amount DOUBLE PRECISION,
    pay_amount DOUBLE PRECISION,
    madeby TEXT,
    voucher TEXT,
    balance DOUBLE PRECISION,
    is_frozen BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS serving_booking_key_index ON serving_booking_rooms (ref, property_name, arrival_date, departure_date);
CREATE INDEX IF NOT EXISTS serving_frozen_booked_on_idx ON serving_booking_rooms ("booked on") WHERE is_frozen = true;