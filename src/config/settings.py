from pathlib import Path

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

MOTEL_DIRECT_CHANNEL = ['staffs', 'luo', 'alanaga', '136bealeyairbnb1', '136bealeyreconlin']

BOOKING_KEY = ['ref', 'is_136_motel', 'arrival_date', 'departure_date']

MODEL_PARAMS = {
    "random_forest": {
        'n_jobs': -2,
        'random_state': 67,
        'class_weight': 'balanced',
    },
    "xgboost": {
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
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 3],
    }
}

MIN_ACCEPTED_PRECISION = 0.73

ARTIFACTS_PATH = PROJECT_ROOT / "src" / "artifacts"
MADEBY_ENCODER_FILE = ARTIFACTS_PATH / "madeby_encoder.joblib"
MODEL_PATH = ARTIFACTS_PATH / "final_model.joblib"

GRID_SEARCH_PARAMS = None

LOAD_OFFLINE_DATA_QUERY = """
                            SELECT resid, ref, book_owner, "booked on", property_name, arrival_date, 
                            departure_date, nights, custid, customer_notes, cust_country, date_cancelled, 
                            status, pax, unit_code, room_code, room_amount, extras_amount, tot_amount, 
                            pay_amount, madeby, voucher, balance FROM offline_bookings WHERE room_code IS NOT NULL
                            """

LOAD_ONLINE_DATA_QUERY = """
                            SELECT resid, ref, book_owner, "booked on", property_name, arrival_date, 
                            departure_date, nights, custid, customer_notes, cust_country, date_cancelled, 
                            status, pax, unit_code, room_code, room_amount, extras_amount, tot_amount, 
                            pay_amount, madeby, voucher, balance FROM online_bookings
                            """