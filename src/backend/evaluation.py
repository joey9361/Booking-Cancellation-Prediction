import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

def evaluate_custom_predictions(
    y_tested: np.ndarray, 
    custom_predictions: np.ndarray
    ) -> tuple[float, float, float, list[list[int]]]:
    # compare test targets and predictions
    f1Scores = round(f1_score(y_tested, custom_predictions, average='weighted'), 3)

  
    recallScore = recall_score(y_tested, custom_predictions, pos_label=1)
    precisionScore = precision_score(y_tested, custom_predictions, pos_label=1)
    confusionMatrix = confusion_matrix(y_tested, custom_predictions)
    return f1Scores, recallScore, precisionScore, confusionMatrix.tolist()