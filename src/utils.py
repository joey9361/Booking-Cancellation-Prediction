import pandas as pd
from config.settings import PROJECT_ROOT, RELEVANT_COLS
from database import Database
from exceptions import CustomException
from logger import logging
import sys

def csv_to_staging_tables(datamanager: Database, csv_path: str, conn) -> None:
    """
    Load raw CSVs into offline staging tables.
    """
    try:
        full_path = PROJECT_ROOT / "data" / csv_path
        df = pd.read_csv(full_path, low_memory=False)
        logging.info(f"Loaded raw data from {full_path}")
        df = df[RELEVANT_COLS].astype(str).copy()
        datamanager.pandas_to_sql(df, "staging_booking_rooms", target_conn=conn)
    except FileNotFoundError as e:
        logging.exception(f"Missing raw data file: %s", full_path)
        raise CustomException(e, sys) from e
    except KeyError as e:
        logging.exception(f"Error aligning raw dataframe to staging dataframe schema: %s", full_path)
        raise CustomException(e, sys) from e
    except Exception as e:
        logging.exception(f"Error loading raw data file: %s", full_path)
        raise CustomException(e, sys) from e

