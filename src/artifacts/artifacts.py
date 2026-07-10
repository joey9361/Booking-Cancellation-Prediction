from pathlib import Path
from config.settings import ARTIFACTS_PATH
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from exceptions import CustomException
from logger import logging
import sys
import json
from datetime import datetime

ROOM_CODE_LOOKUP_FILE = "room_code_lookup.joblib"
HISTORICAL_RATES_FILE = "historical_room_rates.joblib"
MADEBY_ENCODER_FILE = "madeby_encoder.joblib"

def save_joblib_artifacts(
    filepath: Path,
    artifact: pd.DataFrame | OneHotEncoder
) -> None:
    path = ARTIFACTS_PATH
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filepath
    joblib.dump(artifact, file_path)
    logging.info(f"Artifact saved to {file_path}")

def load_joblib_artifacts(filepath: Path) -> pd.DataFrame | OneHotEncoder:
    try:
        artifact = joblib.load(filepath)
        logging.info(f"Joblib artifacts loaded from {filepath}")
        return artifact
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise CustomException(f"File not found: {e}", sys)
    except Exception as e:
        logging.error(f"Error loading joblib artifacts: {e}")
        raise CustomException(f"Error loading joblib artifacts: {e}", sys)

def save_preprocessing_artifacts(
    room_code_lookup: pd.DataFrame,
    historical_room_rates: pd.DataFrame,
    OH_encoder: OneHotEncoder
) -> None:
    save_joblib_artifacts(ROOM_CODE_LOOKUP_FILE, room_code_lookup)
    save_joblib_artifacts(HISTORICAL_RATES_FILE, historical_room_rates)
    save_joblib_artifacts(MADEBY_ENCODER_FILE, OH_encoder)
    logging.info(f"Preprocessing artifacts saved")

def load_preprocessing_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    return (
        load_joblib_artifacts(ARTIFACTS_PATH / ROOM_CODE_LOOKUP_FILE), # room code lookup table
        load_joblib_artifacts(ARTIFACTS_PATH / HISTORICAL_RATES_FILE), # historical room rates lookup table
        load_joblib_artifacts(ARTIFACTS_PATH / MADEBY_ENCODER_FILE) # OneHotEncoder for madeby column
    )

def _save_json_artifacts(
    filepath: str,
    artifacts: dict
) -> None:
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    full_path = ARTIFACTS_PATH / filepath
    with open(full_path, 'w') as f:
        json.dump(artifacts, f)
    logging.info(f"Artifacts saved to {full_path}")
    
def save_threshold_artifacts(
    precision_vals: list,
    recall_vals: list,
    threshold_vals: list
) -> None:
    if not precision_vals:
        raise CustomException(f"Precision values not found in threshold artifacts", sys)
    if not recall_vals:
        raise CustomException(f"Recall values not found in threshold artifacts", sys)
    if not threshold_vals:
        raise CustomException(f"Threshold values not found in threshold artifacts", sys)
    if not (len(precision_vals) == len(recall_vals) == len(threshold_vals)):
        raise CustomException(f"""Precision, recall, and threshold values must have the same length, 
                                got {len(precision_vals)}, {len(recall_vals)}, {len(threshold_vals)}""", sys)

    logging.info(f"Precision, recall, and threshold values have the same length and correct type")

    threshold_artifacts = {
        "precision_vals": precision_vals,
        "recall_vals": recall_vals,
        "threshold_vals": threshold_vals
    }

    _save_json_artifacts("threshold_artifacts.json", threshold_artifacts)
    logging.info(f"Threshold artifacts saved")


def save_prediction_artifacts(
    best_threshold: float,
    min_accepted_precision: float,
    train_inference_time_cutoff: datetime
    ) -> None:
    try:
        prediction_artifacts = {
            "best_threshold": float(best_threshold),
            "min_accepted_precision": float(min_accepted_precision),
            "train_inference_time_cutoff": train_inference_time_cutoff.isoformat()
        }
        _save_json_artifacts("prediction_artifacts.json", prediction_artifacts)
        logging.info(f"Prediction artifacts saved")
    except TypeError as e:
        logging.error(f"Must be a float, got {type(best_threshold)}: {e}")
        raise CustomException(f"Type error: {e}", sys)
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise CustomException(f"Value error: {e}", sys)
    except Exception as e:
        logging.exception(f"Error saving prediction artifacts: %s", e)
        raise CustomException(f"Error saving prediction artifacts: {e}", sys)

def load_json_artifacts(filepath: Path) -> dict:
    try:
        with open(filepath, 'r') as f:
            artifacts = json.load(f)
        logging.info(f"JSON artifacts loaded from {filepath}")
        return artifacts
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise CustomException(f"File not found: {e}", sys)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        raise CustomException(f"Error decoding JSON: {e}", sys)
    except Exception as e:
        logging.error(f"Error loading JSON artifacts: {e}")
        raise CustomException(f"Error loading JSON artifacts: {e}", sys)