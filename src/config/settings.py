from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Columns to drop from dataframe
USELESS_COLS = [
    "comments", "notes", "room_notes", "extra_notes", "other_info", "veh_reg",
    "cust ref", "repeat_guest", "linen", "please select a site", "milk",
    "purpose of visit", "do you require a gst invoice to be issued?",
    "what will you sleep in?", "length of campervan", "dog registration",
    "mode of your site", "please enter the length (in metres) of chosen mode",
    "please select your mode of camping", "milk preference", "customer notes",
    "fly buys card ", "dog breed", "have you received the covid-19 vaccination?",
    "date of birth", "cust_postcode", "cust_city", "cust_suburb", "cust_street", "cust_fax",
    "cust_email", "cust_fname", "cust_lname", "cust_title", "cust_phone",
    "pid", "user_id", "cust_newsletter", "cust_balance", "misc_amount",
    "inv_amount",
]

RELEVANT_COLS = ['resid', 'ref', 'book_owner', 'booked on', 'property_name',
       'arrival_date', 'departure_date', 'nights', 'custid', 'customer_notes',
       'cust_country', 'date_cancelled', 'status', 'pax', 'unit_code',
       'room_code', 'room_amount', 'extras_amount', 'tot_amount', 'pay_amount',
       'madeby', 'voucher', 'balance']

MOTEL_DIRECT_CHANNEL = ['staffs', 'luo', 'alanaga', '136bealeyairbnb1', '136bealeyreconlin']

BOOKING_KEY = ['ref', 'is_136_motel', 'arrival_date', 'departure_date']

MODEL_PARAMS = {
    "random_forest": {
        'n_jobs': -2,
        'random_state': 67,
        'class_weight': 'balanced',
    },
    "xgb": {
        "n_jobs": -2,
        "random_state": 67,
        "scale_pos_weight": 1.0,
    }
}

PARAM_TUNING_GRID = {
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 5],
    },
    "xgb": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 3],
    }
}

MIN_ACCEPTED_PRECISION = 0.875

ARTIFACTS_PATH = PROJECT_ROOT / "src" / "artifacts"
MADEBY_ENCODER_FILE = ARTIFACTS_PATH / "madeby_encoder.joblib"
MODEL_PATH = ARTIFACTS_PATH / "final_model.joblib"

GRID_SEARCH_PARAMS = None

LOAD_OFFLINE_DATA_QUERY = """
                            SELECT resid, ref, book_owner, "booked on", property_name, arrival_date, 
                            departure_date, nights, custid, customer_notes, cust_country, date_cancelled, 
                            status, pax, unit_code, room_code, room_amount, extras_amount, tot_amount, 
                            pay_amount, madeby, voucher, balance 
                            FROM serving_booking_rooms 
                            WHERE room_code IS NOT NULL
                            AND is_frozen = true
                            AND "booked on" < :time_cutoff
                            """

LOAD_ONLINE_DATA_QUERY = """
                            SELECT ref, MIN("booked on")::TEXT as "booked on", property_name, 
                            arrival_date::TEXT, departure_date::TEXT, BOOL_OR(is_frozen) as is_frozen,
                            CASE
                                WHEN BOOL_OR(is_frozen) = FALSE THEN NULL
                                WHEN BOOL_OR(date_cancelled IS NOT NULL) THEN TRUE
                                ELSE FALSE
                            END as is_cancelled
                            FROM serving_booking_rooms
                            WHERE "booked on" >= :time_cutoff
                            AND "booked on" < arrival_date
                            GROUP BY ref, property_name, arrival_date, departure_date
                            ORDER BY MIN("booked on"::TIMESTAMP) DESC
                        """

FETCH_INFERENCE_ROWS_QUERY = """
                            SELECT resid, ref, book_owner, "booked on", property_name, arrival_date, 
                            departure_date, nights, custid, customer_notes, cust_country, date_cancelled, 
                            status, pax, unit_code, room_code, room_amount, extras_amount, tot_amount, 
                            pay_amount, madeby, voucher, balance, is_frozen
                            FROM serving_booking_rooms
                            WHERE ref = :ref
                            AND property_name = :property_name
                            AND arrival_date::TEXT = :arrival_date
                            AND departure_date::TEXT = :departure_date
                            """



CREATE_SCHEMA_SQL_PATHS = [
    PROJECT_ROOT / "sql" / "create_staging.sql",
    PROJECT_ROOT / "sql" / "create_serving.sql",
    PROJECT_ROOT / "sql" / "create_finals.sql",
]

PREPROCESSING_SQL_PATHS = [
    PROJECT_ROOT / "sql" / "validation.sql",
    PROJECT_ROOT / "sql" / "merge_finals_to_serving.sql"
]

