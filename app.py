from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.artifacts.artifacts import load_json_artifacts, load_preprocessing_artifacts, load_joblib_artifacts
from backend.model import apply_best_threshold
from backend.preprocessing import run_online_preprocessing
from src.config.settings import ARTIFACTS_PATH, MODEL_PATH
from pydantic import BaseModel, Field, ConfigDict
from src.exceptions import CustomException
from src.logger import logging
from dotenv import load_dotenv

load_dotenv()

class BookingRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    resid: int
    ref: int
    book_owner: str
    booked_on: str = Field(alias="booked on")
    property_name: str
    arrival_date: str
    departure_date: str
    nights: int
    pax: int 
    custid: str | None
    customer_notes: str | None
    cust_country: str | None
    date_cancelled: str | None
    status: str | None
    unit_code: str | None
    room_code: str | None
    room_amount: float | None
    extras_amount: float | None
    tot_amount: float | None
    pay_amount: float | None
    madeby: str | None
    voucher: str | None
    balance: float | None


class Prediction(BaseModel):
    will_cancel: bool
    confidence_probability: float

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
    logging.info("Model unloaded")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health_check():
    pass

@app.get("/prediction-artifacts")
def get_prediction_artifacts():
    return {
        "best_threshold": float(app.state.prediction_artifacts['best_threshold']),
        "min_accepted_precision": float(app.state.prediction_artifacts['min_accepted_precision'])
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
def predict_booking_cancellation(booking_rows: list[BookingRow], threshold: float | None = None):
    """Inference prediction for a single booking with one or multiple room level rows of data"""
    if threshold is None:
        threshold = app.state.prediction_artifacts['best_threshold']
    # convert pydantic objects for each row to dicts
    booking = [row.model_dump(by_alias=True) for row in booking_rows]
    room_code_lookup, room_rate_lookup, OH_encoder = app.state.preprocessing_artifacts
    X = run_online_preprocessing(booking, room_code_lookup, room_rate_lookup, OH_encoder)
    # get model
    model = app.state.model
    prediction, confidence = apply_best_threshold(model, X, threshold)
    return Prediction(
        will_cancel=prediction[0], 
        confidence_probability=confidence[0])

@app.get('/database-bookings-inference')
def database_bookings_inference():
    pass

