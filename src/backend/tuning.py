from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from typing import Iterable
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from config.settings import MIN_ACCEPTED_PRECISION

def custom_temporal_cv(
    X_df: pd.DataFrame,
    temporal_cv_params: dict | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Creates expanding-window time-series CV splits.

    X_df must already be sorted by booking time (e.g. booked_on) ascending.
    Each fold trains on past rows and validates on the next chronological block.
    """
    tscv = TimeSeriesSplit(**(temporal_cv_params or {"n_splits": 5}))
    return list(tscv.split(X_df))

def custom_hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    param__tuning_grid: dict,
    estimator: BaseEstimator,
    scoring: str | callable = "f1",
    cv_splits: int | Iterable | None = None,
    grid_search_params: dict | None = None,
) -> BaseEstimator:
    """Runs grid search with temporal CV by default."""
    if cv_splits is None:
        cv_splits = custom_temporal_cv(X)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param__tuning_grid,
        scoring=scoring,
        cv=cv_splits,
        **(grid_search_params or {}),
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_
    

def find_best_threshold(
    probabilities: np.ndarray, 
    Y_val: pd.Series):
    # probability threshold tuning
    y_probs = probabilities[:, 1]
    # find precision recalls for each threshold
    precision, recall, threshold = precision_recall_curve(Y_val, y_probs)
    # find best threshold emphasising higher recall
    precision = precision[:-1]
    recall = recall[:-1]

    precision_thresholds = np.where(precision > MIN_ACCEPTED_PRECISION)[0]

    if len(precision_thresholds) > 0:
        best_index = precision_thresholds[np.argmax(recall[precision_thresholds])]
        best_threshold = threshold[best_index]
    else:
        best_threshold = 0.5
    return best_threshold, precision.tolist(), recall.tolist(), threshold.tolist()