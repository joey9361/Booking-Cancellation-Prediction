import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30


def _get_json(path: str) -> dict | list:
    response = requests.get(
        f"{API_BASE_URL}{path}",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def setup_session_state() -> None:
    """Fetch inference bookings and threshold config from the API into session state."""
    st.session_state.inference_bookings = _get_json("/database-bookings-inference")

    prediction_artifacts = _get_json("/prediction-artifacts")
    st.session_state.best_threshold = prediction_artifacts["best_threshold"]
    st.session_state.min_accepted_precision = prediction_artifacts["min_accepted_precision"]

    threshold_response = requests.get(
        f"{API_BASE_URL}/threshold-artifacts",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if threshold_response.status_code == 404:
        st.session_state.threshold_artifacts = None
        st.session_state.custom_threshold_slider_enabled = False
    else:
        threshold_response.raise_for_status()
        st.session_state.threshold_artifacts = threshold_response.json()
        st.session_state.custom_threshold_slider_enabled = True

    st.session_state.startup_complete = True


def main() -> None:
    st.set_page_config(page_title="Booking Cancellation Prediction", layout="wide")

    if not st.session_state.get("startup_complete"):
        with st.spinner("Loading bookings and model settings..."):
            try:
                setup_session_state()
            except requests.RequestException as exc:
                st.error(
                    f"Could not reach the API at {API_BASE_URL}. "
                    "Start the server with `uvicorn app:app --reload` and try again."
                )
                st.exception(exc)
                st.stop()

    st.title("Booking Cancellation Prediction")
    st.caption(f"{len(st.session_state.inference_bookings)} bookings available for inference.")


if __name__ == "__main__":
    main()
