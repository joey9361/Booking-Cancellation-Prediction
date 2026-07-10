import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from src.logger import logging

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30


def _get_json(path: str) -> dict | list:
    """Get the json response from a GET api endpoint"""
    response = requests.get(
        f"{API_BASE_URL}{path}",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _sync_precision_to_recall() -> None:
    if st.session_state.get("_syncing_pr_sliders"):
        return
    st.session_state._syncing_pr_sliders = True
    st.session_state.recall_idx = st.session_state.precision_idx
    st.session_state._syncing_pr_sliders = False


def _sync_recall_to_precision() -> None:
    if st.session_state.get("_syncing_pr_sliders"):
        return
    st.session_state._syncing_pr_sliders = True
    st.session_state.precision_idx = st.session_state.recall_idx
    st.session_state._syncing_pr_sliders = False


def render_threshold_controls() -> None:
    """Render the precision recall value sliders which are synced up 
    together to display their coupled values given by their respective lists 
    called through api at startup and stored in session state. A threshold value will
    be displayed showing the respective threshold value required to achieve the precision, 
    recall value shown by the slider.
    """
    threshold_artifacts = st.session_state.get("threshold_artifacts")
    if not threshold_artifacts:
        st.info("Threshold tuning artifacts are unavailable for this model run.")
        return

    precision_vals = threshold_artifacts.get("precision_vals", [])
    recall_vals = threshold_artifacts.get("recall_vals", [])
    threshold_vals = threshold_artifacts.get("threshold_vals", [])
    # verify that lengths of precision, recall, threshold list values are equal and not corrupted
    lengths_match = len(precision_vals) == len(recall_vals) == len(threshold_vals)
    if not lengths_match or not precision_vals:
        st.warning("Threshold artifacts are incomplete. Re-run threshold tuning.")
        return
    # initialize default slider values at startup in session state
    max_idx = len(precision_vals) - 1
    if "precision_idx" not in st.session_state:
        st.session_state.precision_idx = 0
    if "recall_idx" not in st.session_state:
        st.session_state.recall_idx = st.session_state.precision_idx
    if "_syncing_pr_sliders" not in st.session_state:
        st.session_state._syncing_pr_sliders = False

    st.subheader("Precision / Recall / Threshold Coupled Slider")
    st.caption(
        "Sliders share the same index. Precision increases left-to-right; "
        "recall is mirrored so equal thumb distance from each end."
    )
    # create precision recall sliders by using 
    st.select_slider(
        "Precision",
        options=list(range(max_idx + 1)),
        value=st.session_state.precision_idx,
        format_func=lambda idx: f"{precision_vals[idx]:.3f}",
        key="precision_idx",
        on_change=_sync_precision_to_recall,
    )

    st.select_slider(
        "Recall",
        options=list(range(max_idx, -1, -1)),
        value=st.session_state.recall_idx,
        format_func=lambda idx: f"{recall_vals[idx]:.4f}",
        key="recall_idx",
        on_change=_sync_recall_to_precision,
    )
    # display the threshold value based on what the current precision 
    # recall slider values are set to via the same index
    selected_idx = st.session_state.precision_idx
    current_threshold = float(threshold_vals[selected_idx])
    st.metric("Selected Threshold", f"{current_threshold:.4f}")
    st.caption(
        "Index "
        f"{selected_idx}: "
        f"precision={precision_vals[selected_idx]:.4f}, "
        f"recall={recall_vals[selected_idx]:.4f}, "
        f"threshold={current_threshold:.4f}"
    )
    # saves the current displayed threshold value if button pushed
    save_col, reset_col = st.columns(2)
    with save_col:
        if st.button("Save threshold for prediction", use_container_width=True):
            st.session_state.prediction_threshold = current_threshold
    # resets the current displayed threshold value if button pushed
    with reset_col:
        if st.button("Reset to default threshold", use_container_width=True):
            st.session_state.prediction_threshold = None

    # display the threshold value that is currently saved and ready to use for prediction
    saved_threshold = st.session_state.get("prediction_threshold")
    if saved_threshold is None:
        default_threshold = float(st.session_state.get("best_threshold", current_threshold))
        st.caption(f"Prediction will use the default threshold ({default_threshold:.4f}).")
    else:
        st.caption(f"Saved prediction threshold: {float(saved_threshold):.4f}")


def _bookings_to_display_df(bookings: list[dict]) -> pd.DataFrame:
    """Convert the bookings list to a pandas dataframe for proper formatting and display in the table"""
    rows = []
    for booking in bookings:
        booked_on = booking.get("booked on") or booking.get("booked_on")
        is_frozen = bool(booking.get("is_frozen"))
        rows.append(
            {
                "Booked on": booked_on,
                "Inactive": "✓" if is_frozen else "",
                "Ref": booking.get("ref"),
                "Property": booking.get("property_name"),
                "Arrival": booking.get("arrival_date"),
                "Departure": booking.get("departure_date"),
            }
        )
    return pd.DataFrame(rows)


def _build_prediction_payload(
    booking: dict,
    prediction_threshold: float | None,
) -> dict:
    """Build request payload for POST /predict-booking-cancellation."""
    payload = {
        "json": {
            "ref": booking["ref"],
            "property_name": booking["property_name"],
            "arrival_date": booking["arrival_date"],
            "departure_date": booking["departure_date"],
        },
        "params": {},
    }
    if prediction_threshold is not None:
        payload["params"]["threshold"] = float(prediction_threshold)
    return payload


def _run_prediction(payload: dict) -> dict:
    """Run the prediction for a given booking using the POST /predict-booking-cancellation endpoint"""
    response = requests.post(
        f"{API_BASE_URL}/predict-booking-cancellation",
        json=payload["json"],
        params=payload["params"],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _display_prediction_result(result: dict) -> None:
    will_cancel = bool(result["will_cancel"])
    confidence = float(result["confidence_probability"])
    if result["actual_target"] is None:
        actual_target = None
    else:
        actual_target = bool(result["actual_target"])

    st.subheader("Prediction Result")
    label_col, value_col, target_col = st.columns(3)
    with label_col:
        st.metric("Outcome", "Will cancel" if will_cancel else "Will not cancel")
    with value_col:
        st.metric("Cancellation Confidence", f"{confidence * 100:.1f}%")
    with target_col:
        if actual_target is not None:
            st.metric("Actual Target", "Cancelled" if actual_target else "Not cancelled")
        else:
            st.metric("Actual Target", "Unknown")


def render_inference_bookings() -> None:
    """
    Render the inference bookings table and allow the user to select a booking to predict cancellation for.
    """
    bookings = st.session_state.get("inference_bookings", [])
    st.subheader("Inference Bookings")
    st.caption("✓ indicates an inactive (frozen) booking. Select a row to choose a booking.")

    if not bookings:
        st.info("No inference bookings are available.")
        st.session_state.selected_booking = None
        return
    # render the bookings table
    display_df = _bookings_to_display_df(bookings)
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="inference_bookings_table",
    )
    # set the selected row index in session state
    selected_rows = selection.selection.rows if selection.selection else []
    if selected_rows:
        selected_idx = selected_rows[0]
        st.session_state.selected_booking = bookings[selected_idx]     
    elif "selected_booking" not in st.session_state:
        st.session_state.selected_booking = None

    if st.button("Predict cancellation", use_container_width=True):
        selected_booking = st.session_state.get("selected_booking")
        if not selected_booking:
            st.warning("Please select a booking first.")
        else:
            payload = _build_prediction_payload(
                selected_booking,
                st.session_state.get("prediction_threshold"),
            )
            st.session_state.prediction_payload = payload
            try:
                with st.spinner("Running prediction..."):
                    st.session_state.prediction_result = _run_prediction(payload)
            except requests.HTTPError as exc:
                st.session_state.prediction_result = None
                detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                logging.exception("Prediction failed: %s", exc)
                st.error(f"Prediction failed: {detail}")
            except requests.RequestException as exc:
                st.session_state.prediction_result = None
                logging.exception("Could not reach the API at %s. Error: %s", API_BASE_URL, exc)
                st.error(f"Could not reach the API at {API_BASE_URL}.")
                st.exception(exc)

    if st.session_state.get("prediction_result"):
        _display_prediction_result(st.session_state.prediction_result)


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

    st.session_state.prediction_threshold = None
    st.session_state.prediction_payload = None
    st.session_state.prediction_result = None
    st.session_state.selected_booking = None
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
    render_threshold_controls()
    render_inference_bookings()


if __name__ == "__main__":
    main()
