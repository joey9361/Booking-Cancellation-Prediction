from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.artifacts.artifacts import load_json_artifacts, load_preprocessing_artifacts, load_joblib_artifacts
from src.backend.model import apply_best_threshold
from src.backend.preprocessing import run_online_preprocessing
from src.config.settings import ARTIFACTS_PATH, MODEL_PATH, LOAD_ONLINE_DATA_QUERY, FETCH_INFERENCE_ROWS_QUERY
from pydantic import BaseModel, Field, ConfigDict
from src.exceptions import CustomException
from src.logger import logging
from dotenv import load_dotenv
from src.database import create_datamanager
import os
from datetime import datetime

load_dotenv()

class BookingRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    ref: str
    property_name: str
    arrival_date: str
    departure_date: str
    # Only required for inference
    booked_on: str | None = Field(alias="booked on", default=None) 
    is_frozen: bool | None = None
    is_cancelled: bool | None = None

class Prediction(BaseModel):
    will_cancel: bool
    confidence_probability: float
    actual_target: bool | None = None

class ThresholdArtifacts(BaseModel):
    precision_vals: list[float]
    recall_vals: list[float]
    threshold_vals: list[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = MODEL_PATH
    if not model_path:
        raise RuntimeError("MODEL_PATH is not set in settings.py")
    try:
        app.state.datamanager = create_datamanager()
        logging.info(f"Datamanager created successfully")
        app.state.model = load_joblib_artifacts(model_path)
        logging.info(f"Model loaded successfully")
        app.state.prediction_artifacts = load_json_artifacts(
                                                ARTIFACTS_PATH.joinpath("prediction_artifacts.json")
                                        )
        logging.info(f"Prediction artifacts loaded successfully")
        app.state.preprocessing_artifacts = load_preprocessing_artifacts()
        room_code_lookup, room_rate_lookup, OH_encoder = app.state.preprocessing_artifacts
        if (
            room_code_lookup is None or room_code_lookup.empty 
            or room_rate_lookup is None or room_rate_lookup.empty 
            or OH_encoder is None or not OH_encoder.get_feature_names_out().any()
            ):
            raise ValueError("""Preprocessing artifacts incomplete: missing or 
                                empty room code lookup or room rate lookup or OH encoder""") 
        logging.info(f"Preprocessing artifacts loaded successfully")
    except Exception as e:
        logging.error(f"Error loading artifacts: {e}")
        raise RuntimeError(f"Error loading artifacts: {e}") from e
    yield
    app.state.model = None
    app.state.prediction_artifacts = None
    app.state.preprocessing_artifacts = None
    app.state.datamanager = None
    logging.info("Model unloaded")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health_check():
    pass

@app.get("/prediction-artifacts")
def get_prediction_artifacts():
    return {
        "best_threshold": float(app.state.prediction_artifacts['best_threshold']),
        "min_accepted_precision": float(app.state.prediction_artifacts['min_accepted_precision']),
        "train_inference_time_cutoff": app.state.prediction_artifacts['train_inference_time_cutoff']
    }

@app.get("/threshold-artifacts", response_model=ThresholdArtifacts)
def get_threshold_artifacts():
    """Called once at streamlit app startup and persisted in session state"""
    threshold_artifacts_path = ARTIFACTS_PATH.joinpath("threshold_artifacts.json")
    if not threshold_artifacts_path.exists():
        logging.info(f"Threshold artifacts not found, tuning wasn't performed, don't display custom threshold slider")
        raise HTTPException(
            status_code=404,
            detail="Train with threshold_tuning=True to enable PR curve sliders."
        )
    threshold_artifacts = load_json_artifacts(ARTIFACTS_PATH.joinpath("threshold_artifacts.json"))
    for key in ("precision_vals", "recall_vals", "threshold_vals"):
        if key not in threshold_artifacts or not threshold_artifacts[key]:
            logging.error(f"{key} not found in threshold artifacts")
            raise HTTPException(
                status_code=404,
                detail=f"Threshold artifacts incomplete: missing or empty '{key}'. Don't display custom threshold slider.",
            )
    logging.info(f"Threshold artifacts loaded successfully, passing to streamlit session state now")
    return ThresholdArtifacts(**threshold_artifacts)

@app.post("/predict-booking-cancellation", response_model=Prediction)
def predict_booking_cancellation(booking_rows: BookingRow, threshold: float | None = None):
    """Inference prediction for a single booking with one or multiple room level rows of data"""
    try:
        if threshold is None:
            threshold = app.state.prediction_artifacts['best_threshold']
        # fetch inference rows from database
        booking = app.state.datamanager.load_query(
            FETCH_INFERENCE_ROWS_QUERY, 
            params={
                'ref': booking_rows.ref, 
                'property_name': booking_rows.property_name, 
                'arrival_date': booking_rows.arrival_date, 
                'departure_date': booking_rows.departure_date
                }
            )
        if booking.empty:
            logging.error(f"Booking not found in database via booking keys: {booking_rows.model_dump(by_alias=True)}")
            raise HTTPException(
                status_code=404,
                detail=f"Booking not found in database, try selecting a different booking"
            )
        logging.info(f"Booking fetched successfully from database")
        room_code_lookup, room_rate_lookup, OH_encoder = app.state.preprocessing_artifacts
        X = run_online_preprocessing(booking, room_code_lookup, room_rate_lookup, OH_encoder)
        logging.info(f"Online preprocessing completed successfully")
        # get model
        model = app.state.model
        prediction, confidence = apply_best_threshold(model, X, threshold)
        logging.info(f"Prediction completed successfully")
        # get actual target variable if booking is inactive
        actual_target = bool(booking['date_cancelled'].notnull()[0]) if booking['is_frozen'][0] else None
        return Prediction(
                will_cancel=prediction[0], 
                confidence_probability=confidence[0],
                actual_target=actual_target)
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception(f"Error predicting booking cancellation: %s", e)
        raise RuntimeError(f"Error predicting booking cancellation: {e}") from e

@app.get('/database-bookings-inference', response_model=list[BookingRow])
def database_bookings_inference():
    """
    Query the database at streamlit startup to get the booking data options 
    for inference given a set time cutoff
    """
    try:
        train_inference_time_cutoff = app.state.prediction_artifacts['train_inference_time_cutoff']
        df = app.state.datamanager.load_query(
            LOAD_ONLINE_DATA_QUERY, 
            params={'time_cutoff': datetime.fromisoformat(train_inference_time_cutoff)})
        logging.info(f"Bookings fetched successfully from database")
        return [BookingRow.model_validate(row) for row in df.to_dict(orient='records')]
    except Exception as e:
        logging.error(f"Error fetching bookings from database: {e}")
        raise RuntimeError(f"Error fetching bookings from database: {e}") from e
    
if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)

    uvicorn.run(
        "app:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", "8000")),
        reload=False,
    ) 
