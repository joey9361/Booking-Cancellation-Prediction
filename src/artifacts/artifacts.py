from pathlib import Path
from src.config.pathing import PROJECT_ROOT
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.exceptions import CustomException
import sys
import json

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

    
def save_prediction_artifacts(filedir: Path, threshold: float, model_path: str) -> None:
    filedir.mkdir(parents=True, exist_ok=True)
    if type(float(threshold)) != float:
        raise TypeError(f"Threshold must be a float, got {type(threshold)}")
    threshold_artifacts = {
        "threshold": float(threshold),
        "model_path": str(model_path)
    }
    filepath = filedir / "prediction_artifacts.json"
    with open(filepath, 'w') as f:
        json.dump(threshold_artifacts, f)

def load_saved_prediction_artifacts(filepath: Path | None) -> dict:
    if filepath is None:
        filepath = ARTIFACTS_DIR / "prediction_artifacts.json"
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found")
    with open(filepath, 'r') as f:
        prediction_artifacts = json.load(f)
    if not isinstance(prediction_artifacts["threshold"], float):
        raise TypeError(f"Threshold must be a float, got {type(prediction_artifacts["threshold"])}")
    if not isinstance(prediction_artifacts["model_path"], str):
        raise TypeError(f"Model path must be a string, got {type(prediction_artifacts["model_path"])}")
    if not Path(prediction_artifacts["model_path"]).exists():
        raise FileNotFoundError(f"Model path {prediction_artifacts["model_path"]} not found")
    return prediction_artifacts