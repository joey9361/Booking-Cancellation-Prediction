DROP TABLE IF EXISTS staging_booking_rooms;

CREATE TABLE staging_booking_rooms (
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
    balance TEXT
);
