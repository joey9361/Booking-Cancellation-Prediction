from utils import csv_to_staging_tables
from testing_database import create_datamanager, Database
from config.settings import PROJECT_ROOT, CREATE_SCHEMA_SQL_PATHS, PREPROCESSING_SQL_PATHS
from exceptions import CustomException
from logger import logging
import sys
from dotenv import load_dotenv

load_dotenv()

def ingest_new_data_pipeline(csv_path: str) -> None:
    """Full pipeline which ingests a csv of data and runs 
    the full preprocessing stages before dumping in serving table"""
    try:
        datamanager = create_datamanager()
        with datamanager.transaction() as conn:
            # Create schema tables
            _read_and_execute_sql_script(datamanager, CREATE_SCHEMA_SQL_PATHS, conn)
            logging.info(f"Schema tables created")
            full_path = PROJECT_ROOT / "data" / csv_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            # Load raw csv data into staging tables
            csv_to_staging_tables(datamanager, csv_path, conn)
            logging.info(f"Loaded raw csv data into staging tables")
            # Execute database preprocessing scripts
            _read_and_execute_sql_script(datamanager, PREPROCESSING_SQL_PATHS, conn)
            logging.info(f"Database preprocessing scripts executed")
    except Exception as e:
        logging.error(f"Error in ingest_new_data_pipeline: {e}")
        logging.info('rolled back ingest_new_data_pipeline transaction')
        raise CustomException(e, sys) from e
    
def _read_and_execute_sql_script(datamanager: Database, sql_paths: list[str], conn) -> None:
    try:
        for path in sql_paths:
            with open(path, 'r') as file:
                sql_script = file.read()
                datamanager.execute_script(sql_script, conn=conn)
                logging.info(f"Executed {path}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {path}")
        raise CustomException(f"File not found: {path}", sys) from e
    except Exception as e:
        logging.error(f"Error in _read_and_execute_sql_script: {e}")
        raise CustomException(e, sys) from e




