def run_ingest():
    from argparse import ArgumentParser
    import sys
    import logging
    from src.ingest import ingest_new_data_pipeline
    
    parser = ArgumentParser()
    parser.add_argument("csv_path", type=str, help="CSV file name to ingest")
    args = parser.parse_args()
    try:
        ingest_new_data_pipeline(args.csv_path)
        logging.info(f"Ingested new data pipeline successfully")
        print("Ingested new data pipeline successfully")
    except Exception as e:
        logging.error(f"Error in ingest_new_data_pipeline: {e}")
        print(f"Error in ingest_new_data_pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_ingest()
    