# Booking cancellation prediction

End-to-end ML project for predicting whether a motel booking will cancel before arrival. Uses real exported motel data over the past 10 years from two motels in New Zealand and designed for continuous data batch processing as new bookings are created. Data is ingested and validated in PostgreSQL, transformed and trained in Python, served with FastAPI, and used from a Streamlit dashboard.

## Disclaimer 

This project was built on real motel PMS booking exports under internal permission.
Raw guest data is not included in the repository for privacy reasons.
The code, SQL pipeline, training/serving architecture, and dashboard are fully documented.
Demo screenshots / walkthrough available on request.

## What this project does

Stack: Python | PostgreSQL | Scikit-learn | XGBoost | FastAPI | Streamlit | Pandas | Numpy

- Ingests booking CSV exports into PostgreSQL (staging -> finals -> serving)
- Trains a cancellation model from frozen historical bookings
- Serves predictions through FastAPI
- Lets users explore precision/recall/threshold trade-offs in Streamlit
- Supports booking selection and custom-threshold prediction from the dashboard

## Quick project summary

- **Task:** binary classification (`will_cancel`)
- **Models used:** Random Forest and XGBoost (**default: XGBoost**)
- **Feature pipeline:** room-level rows transformed to booking-level features
- **Serving split:** training on frozen past bookings, inference on recent valid bookings
  - (This prevents future edits to a booking from leaking into the training labels, mimicking real-world inference conditions).


## Model metrics

Final metrics from the held-out test split  
(default pipeline: **XGBoost** + hyperparameter tuning + PR threshold tuning).

| Model | Precision | Recall | Weighted F1 | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---|
| XGBoost (final) | 0.8135 | 0.9213 | 0.961 | `[[2646, 94], [35, 410]]` |

Threshold settings from that run:
- `MIN_ACCEPTED_PRECISION = 0.875`
- `best_threshold ≈ 0.683`

Notes:
- Threshold tuning targeted confirmation precision while keeping recall around ~0.92.
- The dashboard still lets you explore other precision/recall trade-offs without retraining.

## Pipeline overview

**Offline (ingest + training)**

```text
CSV (data/bookings.csv)
   -> SQL validation + merge (staging/finals/serving)
   -> offline preprocessing + feature engineering
   -> train/tune model
   -> save artifacts (model + lookups + threshold artifacts)
```

**Online (serving)**

```text
Streamlit UI
   -> FastAPI endpoints
   -> booking-key fetch from serving table
   -> online preprocessing
   -> predict_proba + threshold
   -> result in dashboard (with actual target for frozen bookings)
```

## Key implementation points

- Booking key used for inference fetch: `ref`, `property_name`, `arrival_date`, `departure_date`
- Inference list excludes invalid lead-time rows (`"booked on" < arrival_date`)
- FastAPI loads model/artifacts once at startup (lifespan)
- Streamlit supports:
  - coupled precision/recall sliders
  - custom threshold save/reset
  - booking table selection
  - prediction request + result display

## Limitations

- Cancelled bookings often do not have reliable room-rate values.
- Night extensions can overwrite original stay duration in source records.
- The same physical unit can be sold under different room codes; modal room-code lookup is used.
- Some source rows had booking dates after arrival dates (negative lead time), so these are filtered out.

## Project structure

```text
app.py
streamlit_app.py
pyproject.toml
sql/
src/
  config/settings.py
  ingest.py
  utils.py
  database.py
  cli/
  backend/
  artifacts/
  notebooks/
logs/
```

## Setup (Windows / PowerShell)

### 1) Environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2) Configure `.env`

```env
DBNAME=motel_data
DBUSER=postgres
DBPASSWORD=your_password
host=localhost
port=5432

FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
API_BASE_URL=http://127.0.0.1:8000
DEBUG_MODE=False
```

### 3) Place data

Put a compatible booking export at data/bookings.csv (not included in this repo — see Disclaimer).
Expected columns match RELEVANT_COLS in `src/config/settings.py`.

### 4) Run pipeline

```powershell
ingest_csv bookings.csv
run_main --hyperparameter_tuning --threshold_tuning
python -m app
streamlit run streamlit_app.py
```

`run_main` now defaults to XGBoost. Use `--model random_forest` only if you want the Random Forest variant.

## Operational notes

- After running `run_main` (retraining), restart the FastAPI server (`python -m app`) so it reloads the updated model/artifacts from disk.
- If Streamlit is already open during retraining, refresh/restart the Streamlit page/session so startup-fetched values (booking list, threshold artifacts, default threshold) are refreshed.
- Safe local order is: `ingest_csv` -> `run_main` -> restart API -> restart/refresh Streamlit.

## API endpoints

- `GET /database-bookings-inference`
- `GET /prediction-artifacts`
- `GET /threshold-artifacts`
- `POST /predict-booking-cancellation`

## Reflection

The biggest learning curve was that PMS data is not static. A single booking row can change over time (night extensions, rate changes, status updates), so just training from a raw export can quietly introduce leakage.

I had to design a SQL merge strategy that separates frozen historical records from active ones so training labels stay reliable and inference conditions match real usage.
