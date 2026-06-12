from pathlib import Path
from src.config.pathing import PROJECT_ROOT
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.exceptions import CustomException
import sys

ARTIFACTS_DIR = PROJECT_ROOT / "src" / "artifacts" 

ROOM_CODE_LOOKUP_FILE = "room_code_lookup.joblib"
HISTORICAL_RATES_FILE = "historical_room_rates.joblib"
MADEBY_ENCODER_FILE = "madeby_encoder.joblib"

def save_room_code_lookup(
    lookup: pd.DataFrame
    ) -> None:
    path = ARTIFACTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / ROOM_CODE_LOOKUP_FILE
    joblib.dump(lookup, file_path)


def load_room_code_lookup() -> pd.DataFrame:
    try:
        file_path = ARTIFACTS_DIR / ROOM_CODE_LOOKUP_FILE
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(f"Error loading room code lookup: {e}", sys)

def save_historical_room_rates(
    rates: pd.DataFrame,
) -> None:
    path = ARTIFACTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / HISTORICAL_RATES_FILE
    joblib.dump(rates, file_path)


def load_historical_room_rates() -> pd.DataFrame:
    try:
        file_path = ARTIFACTS_DIR / HISTORICAL_RATES_FILE
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(f"Error loading historical room rates: {e}", sys)


def save_lookup_artifacts(
    room_code_lookup: pd.DataFrame,
    historical_room_rates: pd.DataFrame
) -> None:
    save_room_code_lookup(room_code_lookup)
    save_historical_room_rates(historical_room_rates)

# exclusively for inference
def load_lookup_artifacts(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_room_code_lookup(),
        load_historical_room_rates(),
    )


def save_madeby_encoder(
    encoder: OneHotEncoder
) -> None:
    path = ARTIFACTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / MADEBY_ENCODER_FILE
    joblib.dump(encoder, file_path)


def load_madeby_encoder() -> OneHotEncoder:
    try:
        file_path = ARTIFACTS_DIR / MADEBY_ENCODER_FILE
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(f"Error loading madeby encoder: {e}", sys)

    