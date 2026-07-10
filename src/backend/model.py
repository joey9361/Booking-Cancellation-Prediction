from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from typing import Any
from exceptions import CustomException



def initialize_model(
    model_type: str = "xgb", 
    model_params: dict | None = None
    ) -> Any:
    """Initializes the model with the given type and parameters."""
    if model_type == "random_forest":
        model = RandomForestClassifier(**(model_params or {}))
    elif model_type == "xgb":
        model = XGBClassifier(**(model_params or {}))
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return model

def xgb_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """negatives / positives — same as notebook."""
    y = np.asarray(y)
    positives = (y == 1).sum()
    if positives == 0:
        raise ValueError("Y_train has no positive class")
    return (y == 0).sum() / positives

def apply_best_threshold(
    mymodel: RandomForestClassifier | XGBClassifier, 
    X: pd.DataFrame, 
    best_threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(mymodel, "feature_names_in_"):
        X = X[mymodel.feature_names_in_]
    y_probs = mymodel.predict_proba(X)[:, 1]
    # apply best threshold
    custom_predictions = (y_probs > best_threshold).astype(int)
    return custom_predictions, y_probs