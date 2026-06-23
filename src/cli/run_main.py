def run_main():
    from backend.main import main
    from argparse import ArgumentParser
    import sys
    
    parser = ArgumentParser()
    parser.add_argument(
        "--model", 
        choices=["random_forest", "xgb"], 
        default="random_forest", 
        help="Model type to use"
        )
    parser.add_argument(
        "--hyperparameter_tuning",
        action="store_true",
        help="Whether to perform hyperparameter tuning"
    )
    parser.add_argument(
        "--threshold_tuning",
        action="store_true",
        help="Whether to perform threshold tuning"
    )
    args = parser.parse_args()
    try:
        main(model_type=args.model, 
        hyperparameter_tuning=args.hyperparameter_tuning, 
        threshold_tuning=args.threshold_tuning
        )
        print("Model trained successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_main()
    