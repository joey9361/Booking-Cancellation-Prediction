import pandas as pd
import numpy as np
from src.config.settings import USELESS_COLS, MOTEL_DIRECT_CHANNEL, BOOKING_KEY
from sklearn.preprocessing import OneHotEncoder

# add error handling for inference
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # drop useless columns
    df = df.drop(columns=USELESS_COLS, axis=1, errors='ignore')
    # drop null room_code
    df = df[df['room_code'].notna()]
    # fill null madeby
    df['madeby'] = df['madeby'].fillna('136bealey')
    # replace some madeby categories with 136 on bealey
    df.loc[df['madeby'].isin(MOTEL_DIRECT_CHANNEL), 'madeby'] = '136bealey' 
    return df

def room_level_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # domestic vs international binary encode
    df['is_domestic'] = (df['cust_country'] == 'New Zealand') | (df['cust_country'].isna())
    # is 136 motel binary encode
    df['is_136_motel'] = (df['property_name'] == '136 on Bealey')
    # has customer notes binary encode
    df["has_customer_notes"] = df["customer_notes"].notna()
    # has voucher binary encode
    df["has_voucher"] = df["voucher"].notna()
    # lead time 
    df["booked on"] = pd.to_datetime(df["booked on"], errors="coerce", utc=True, format="mixed")
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce", utc=True, format="mixed")
    df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce", utc=True, format="mixed")
    # lead time days
    df["lead_time"] = df["arrival_date"] - df["booked on"]
    df["lead_time_days"] = round(df["lead_time"].dt.total_seconds() / 86_400, ndigits=2)
    # target column 
    df["is_cancelled"] = df["date_cancelled"].notna()

    return df

# split data into train, validation, test
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    BOOKING_KEY = ['ref', 'is_136_motel', 'arrival_date', 'departure_date']
    """
    # get distinct booking keys in a table
    distinct_bookings = (
        df.groupby(BOOKING_KEY, as_index=False)
        .aggregate(
            booked_on=('booked on', 'min')
        )
    )
    # sort by booked on date
    distinct_bookings = distinct_bookings.sort_values('booked_on', ascending=True)

    # Create train and val set combined
    train_val_cutoff = int(len(distinct_bookings) * 0.8)
    train_val_set = distinct_bookings.iloc[:train_val_cutoff]

    # Finally split into train, validation, test
    val_cutoff = int(len(train_val_set) * 0.8)
    train_set = train_val_set[:val_cutoff]
    validation_set = train_val_set[val_cutoff:]
    test_set = distinct_bookings.iloc[train_val_cutoff:]
    # inner merge distributes the original dataframe rows to the correct set splits
    full_train = df.merge(train_set[BOOKING_KEY], on=BOOKING_KEY, how='inner')
    full_val = df.merge(validation_set[BOOKING_KEY], on=BOOKING_KEY, how='inner')
    full_test = df.merge(test_set[BOOKING_KEY], on=BOOKING_KEY, how='inner')

    return full_train, full_val, full_test

# placeholder room code filter
def _placeholder_room_filter(df: pd.DataFrame) -> pd.Series:
    return (df["room_code"] == "dummy") | (df["room_code"] == "136")

# custom fit and transforms for room_code and room_rate
def fit_room_code(train_df: pd.Dataframe) -> pd.DataFrame:
    """
    Create lookup table with columns [property_name, unit_code, lookup_unit, modal_room_code] 
    using statistics from only a train dataset
    """
    def _lookup_unit(row):
        if row["property_name"] == "136 on Bealey" and row["unit_code"] == "abbey room 12":
            return "Room 12"
        return row["unit_code"]
    # Get the modal room_code of each (property_name, unit_code) type from the training set as a series
    placeholders = _placeholder_room_filter(train_df)
    modal_by_unit = (
    train_df.loc[~placeholders]
    .groupby(["property_name", "unit_code"])["room_code"]
    .apply(lambda s: s.mode().iloc[0])
    .reset_index(name="modal_room_code")
    )

    # Get all distinct (is_136_bealey, unit_code) pairs as rows
    units_table = (
    train_df.loc[train_df["unit_code"].notna(), ["property_name", "unit_code"]]
    .drop_duplicates()
    .reset_index(drop=True)
    )

    units_table["lookup_unit"] = units_table.apply(_lookup_unit, axis=1)

    room_lookup_v2 = units_table.merge(
        modal_by_unit.rename(columns={"unit_code": "lookup_unit"}),
        on=["property_name", "lookup_unit"],
        how="left"
    )
    return room_lookup_v2

def transform_room_code(df: pd.DataFrame, lookup_table: pd.DataFrame) -> None:
    """Replace all dummy room_code for abbey and 136 room_code 
    for 136 on bealey via lookup table created with fit_room_code"""
    placeholder_mask = _placeholder_room_filter(df)
    placeholders = df.loc[placeholder_mask].copy()

    placeholders_impute = placeholders.merge(
        lookup_table, 
        on=['property_name', 'unit_code'], 
        how='left'
        )
    placeholders_impute.index = placeholders.index
    df.loc[placeholder_mask, 'room_code'] = placeholders_impute['modal_room_code']
    df.loc[placeholder_mask & df["room_code"].isna(), "room_code"] = "Unknown"


# rate mask helper function
def _room_rate_filter(df: pd.DataFrame) -> pd.Series:
    return (df['is_cancelled'] == 0) & (df['room_amount'] > 0.0) & (df['nights'] > 0) 

def fit_room_rate(train_df: pd.DataFrame) -> tuple(pd.Series, pd.DataFrame):
    rate_mask = _room_rate_filter(train_df)
    rate_df = train_df.loc[rate_mask, ['ref', 'room_code', 'nights', 'room_amount']].copy()
    rate_df['room_rate'] = rate_df['room_amount'] / rate_df['nights']
    # calculate aggregate average room_rate per room_code
    historical_rates = (rate_df
                        .groupby('room_code')['room_rate']
                        .apply(lambda x: x.median())
                        .reset_index(name="historical_room_rate")
                        )
    # add row 'Unknown' to historical_rates for those with unknown room_code
    historical_rates.loc[len(historical_rates)] = {
        'room_code': 'Unknown', 
        'historical_room_rate': historical_rates['historical_room_rate'].mean()
        }
    return historical_rates

def transform_room_rate(
    df: pd.DataFrame, 
    lookup_table: pd.DataFrame) -> None:
    """
    Calculate the room_rate for each room row with valid room_amount, 
    and impute historical_room_rate via lookup table to those missing a 
    valid room_amount or cancelled rows
    """
    rate_mask = _room_rate_filter(df)
    df['room_rate'] = np.nan
    df.loc[rate_mask, 'room_rate'] = df.loc[rate_mask, 'room_amount'] / df.loc[rate_mask, 'nights']
        
    historical_imputation = df.loc[~rate_mask].merge(lookup_table, on='room_code', how='left')
    historical_imputation.index = df.loc[~rate_mask].index
    df.loc[~rate_mask, 'room_rate'] = historical_imputation['historical_room_rate']

# clean_lead_time
###################################
# add error handling for inference
def _clean_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["lead_time_days"] >= 0].copy()
    df["lead_time_days"] = df["lead_time_days"].clip(upper=525)
    return df

# aggregation to booking level
def booking_level_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate room level rows to booking level using booking key, 
    creating booking level features in the process
    """
    # helpers for aggregation
    df["room_revenue"] = df["room_rate"] * df["nights"]
    df["person_nights"] = df["pax"] * df["nights"]

    # booked on or lead_date needs revisiting for whether should be calculated at room level or booking level df
    # need to figure out how to handle stay duration since not all bookings have a single consistent stay duration for multi-room 
    # decide whether to one hot encode room_code which would add 14 columns eg booked_double_queen
    bookings = df.groupby(['ref', 'is_136_motel', 'arrival_date', 'departure_date'], as_index=False).aggregate(
        madeby=('madeby', 'first'), # booking channel as a string for onehot later
        has_customer_notes=('has_customer_notes', 'first'),
        is_domestic=('is_domestic', 'first'),
        has_voucher=('has_voucher', 'first'),
        total_rooms=('ref', 'size'),
        total_guests=('pax', 'sum'),
        total_room_revenue=('room_revenue', 'sum'), # total booking value from room prices only
        person_nights=('person_nights', 'sum'), # helper, can be deleted for performance
        average_room_rate=('room_rate', 'mean'),
        lead_time_days=('lead_time_days', 'max'),
        is_cancelled=('is_cancelled', 'min') # target column
    )
    # add division by zero error handling
    ########################################################################################
    bookings['average_price_pp'] = bookings['total_room_revenue'] / bookings['person_nights']
    bookings = _clean_lead_time(bookings)
    return bookings

# one hot encode madeby
def custom_OHE(df: pd.DataFrame, column: str, OH_encoder: OneHotEncoder) -> pd.DataFrame:
    ohe_df = pd.DataFrame(
        OH_encoder.transform(df[[column]]),
        columns=OH_encoder.get_feature_names_out([column]),
        index=df.index
    )
    df = pd.concat([df.drop(columns=[column]), ohe_df], axis=1)
    return df

def model_compatible_dfs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_test = df.drop(['ref', 'arrival_date', 'departure_date', 'person_nights'], axis=1)
    X_test = final_test.drop('is_cancelled', axis=1)
    Y_test = final_test['is_cancelled']
    return X_test, Y_test

def run_offline_preprocessing(
    df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Run offline preprocessing on a dataframe, returning the model compatible dfs,
    the fitted room code lookup table, the fitted room rate lookup table, 
    and the fitted one hot encoder as artifacts.

    """
    df = clean_data(df)
    df = room_level_preprocessing(df)
    df = split_data(df) 
    room_code_lookup = fit_room_code(df)
    df = transform_room_code(df, room_code_lookup)
    room_rate_lookup = fit_room_rate(df)
    df = transform_room_rate(df, room_rate_lookup)
    df = booking_level_preprocessing(df)
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(df[['madeby']])
    df = custom_OHE(df, 'madeby', OH_encoder)
    X_test, Y_test = model_compatible_dfs(df)
    return X_test, Y_test, room_code_lookup, room_rate_lookup, OH_encoder

def run_online_preprocessing(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = clean_data(df)
        df = room_level_preprocessing(df)
