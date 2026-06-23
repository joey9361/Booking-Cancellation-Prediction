
CREATE TABLE IF NOT EXISTS finals_booking_rooms (
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

CREATE INDEX IF NOT EXISTS booking_key_index ON finals_booking_rooms (ref, property_name, arrival_date, departure_date);

-- Create a rejected table for rows that failed validation
DROP TABLE IF EXISTS rejected_booking_rooms;
CREATE TABLE rejected_booking_rooms (
    row_id BIGSERIAL PRIMARY KEY,
    resid TEXT ,
    ref TEXT,
    book_owner TEXT,
    "booked on" TEXT,
    property_name TEXT,
    arrival_date TEXT,
    departure_date TEXT,
    nights TEXT,
    custid TEXT,
    customer_notes TEXT,
    cust_country TEXT,
    date_cancelled TEXT,
    "status" TEXT,
    pax TEXT,
    unit_code TEXT,
    room_code TEXT,
    room_amount TEXT,
    extras_amount TEXT,
    tot_amount TEXT,
    pay_amount TEXT,
    madeby TEXT,
    voucher TEXT,
    balance TEXT,
    is_frozen TEXT
);