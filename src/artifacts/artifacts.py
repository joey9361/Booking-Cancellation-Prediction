from pathlib import Path
from src.config.settings import ARTIFACTS_PATH
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.exceptions import CustomException
from src.logger import logging
import sys
import json


ROOM_CODE_LOOKUP_FILE = "room_code_lookup.joblib"
HISTORICAL_RATES_FILE = "historical_room_rates.joblib"
MADEBY_ENCODER_FILE = "madeby_encoder.joblib"

def save_room_code_lookup(
    lookup: pd.DataFrame
    ) -> None:
    path = ARTIFACTS_PATH
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / ROOM_CODE_LOOKUP_FILE
    joblib.dump(lookup, file_path)


def load_room_code_lookup() -> pd.DataFrame:
    try:
        file_path = ARTIFACTS_PATH / ROOM_CODE_LOOKUP_FILE
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(f"Error loading room code lookup: {e}", sys)

def save_historical_room_rates(
    rates: pd.DataFrame,
) -> None:
    path = ARTIFACTS_PATH
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / HISTORICAL_RATES_FILE
    joblib.dump(rates, file_path)


def load_historical_room_rates() -> pd.DataFrame:
    try:
        file_path = ARTIFACTS_PATH / HISTORICAL_RATES_FILE
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
    path = ARTIFACTS_PATH
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / MADEBY_ENCODER_FILE
    joblib.dump(encoder, file_path)


def load_madeby_encoder() -> OneHotEncoder:
    try:
        file_path = ARTIFACTS_PATH / MADEBY_ENCODER_FILE
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(f"Error loading madeby encoder: {e}", sys)

    
def save_prediction_artifacts(
    best_threshold: float, 
    precision_vals: list | None, 
    recall_vals: list | None, 
    threshold_vals: list | None
    ) -> None:
    prediction_artifacts = {}

    if threshold_vals is not None or precision_vals is not None or recall_vals is not None:
        logging.info(f"Threshold tuning was performed")
        if not isinstance(precision_vals, list):
            raise TypeError(f"Precision values must be a list, got {type(precision_vals)}")
        if not isinstance(recall_vals, list):
            raise TypeError(f"Recall values must be a list, got {type(recall_vals)}")
        if not isinstance(threshold_vals, list):
            raise TypeError(f"Threshold values must be a list, got {type(threshold_vals)}")
        if not (len(precision_vals) == len(recall_vals) == len(threshold_vals)):
            raise ValueError(f"""Precision, recall, and threshold values must have the same length, 
                                got {len(precision_vals)}, {len(recall_vals)}, {len(threshold_vals)}""")
        logging.info(f"Precision, recall, and threshold values have the same length and correct type")
        prediction_artifacts = {
            "precision_vals": precision_vals,
            "recall_vals": recall_vals,
            "threshold_vals": threshold_vals
        }

    if type(float(best_threshold)) != float:
        raise TypeError(f"Threshold must be a float, got {type(best_threshold)}")
    logging.info(f"Threshold is a float and correct type")
    prediction_artifacts['best_threshold'] = float(best_threshold)
    
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    filepath = ARTIFACTS_PATH / "prediction_artifacts.json"
    with open(filepath, 'w') as f:
        json.dump(prediction_artifacts, f)
    logging.info(f"Prediction artifacts saved to {filepath}")

def load_saved_prediction_artifacts(filepath: Path | None) -> dict:
    if filepath is None:
        filepath = ARTIFACTS_PATH / "prediction_artifacts.json"
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found")
    with open(filepath, 'r') as f:
        prediction_artifacts = json.load(f)
    return prediction_artifacts