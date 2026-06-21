from src.testing_database import create_datamanager
from preprocessing import run_offline_preprocessing, full_custom_transform
from model import initialize_model, apply_best_threshold, xgb_scale_pos_weight
from tuning import custom_hyperparameter_tuning, custom_temporal_cv, find_best_threshold
from evaluation import evaluate_custom_predictions
from src.config.settings import LOAD_OFFLINE_DATA_QUERY,MODEL_PARAMS, PARAM_TUNING_GRID, GRID_SEARCH_PARAMS, MODEL_PATH, MIN_ACCEPTED_PRECISION, ARTIFACTS_PATH
from joblib import dump
from src.artifacts.artifacts import save_prediction_artifacts, save_threshold_artifacts, load_preprocessing_artifacts   
from src.logger import logging
from src.exceptions import CustomException
import sys
from dotenv import load_dotenv

load_dotenv()

def main(
    model_type: str = 'random_forest', 
    hyperparameter_tuning: bool = True, 
    threshold_tuning: bool = True): # add argument options for tuning the model etc
    # Load the data
    datamanager = create_datamanager()
    df = datamanager.load_query(LOAD_OFFLINE_DATA_QUERY)

    X_train, Y_train, full_val, full_test = run_offline_preprocessing(df)
    room_code_lookup, room_rate_lookup, OH_encoder = load_preprocessing_artifacts()
    X_val, Y_val = full_custom_transform(full_val, room_code_lookup, room_rate_lookup, OH_encoder, is_offline=True)
    X_test, Y_test = full_custom_transform(full_test, room_code_lookup, room_rate_lookup, OH_encoder, is_offline=True)

    # Train the model with default params
    model_params = {**MODEL_PARAMS[model_type]}
    if model_type == "xgboost":
        model_params["scale_pos_weight"] = xgb_scale_pos_weight(Y_train)
    model = initialize_model(model_type, model_params)
    if hyperparameter_tuning:
        # temporal cross validation
        cv_results = custom_temporal_cv(X_train)
        # hyperparameter tuning
        tuned_model = custom_hyperparameter_tuning(
            X_train, 
            Y_train, 
            PARAM_TUNING_GRID[model_type], 
            estimator=model, 
            cv_splits=cv_results,
            grid_search_params=GRID_SEARCH_PARAMS
            )
    else:
        tuned_model = model.fit(X_train, Y_train)
    if threshold_tuning:
        # threshold tuning
        probabilities = tuned_model.predict_proba(X_val)
        best_threshold, precision_vals, recall_vals, threshold_vals = find_best_threshold(probabilities, Y_val)
        min_accepted_precision = MIN_ACCEPTED_PRECISION
        save_threshold_artifacts(precision_vals, recall_vals, threshold_vals)
        # apply best threshold to get custom predictions
        custom_predictions, _ = apply_best_threshold(tuned_model, X_test, best_threshold)
    else:
        # delete existing threshold artifacts if threshold tuning was not performed
        (ARTIFACTS_PATH / "threshold_artifacts.json").unlink(missing_ok=True)
        best_threshold, min_accepted_precision = 0.5, 0.0 # default threshold and precision
        custom_predictions = tuned_model.predict(X_test)
        logging.info(f"Threshold tuning was not performed, using default threshold")
    # evaluate the model
    evaluation_results = evaluate_custom_predictions(Y_test, custom_predictions)
    f1Scores, recallScore, precisionScore, confusionMatrix = evaluation_results
    # save the model and its path
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(tuned_model, MODEL_PATH)
    save_prediction_artifacts(best_threshold, min_accepted_precision)


if __name__ == "__main__":
    from src.testing_database import create_datamanager
    datamanager = create_datamanager()
    main()