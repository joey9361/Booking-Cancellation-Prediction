from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from contextlib import contextmanager
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self, dbname, dbuser, dbpassword, host='localhost', port=5432):
        self.connection_url = URL.create(drivername='postgresql+psycopg2', username=dbuser, password=dbpassword, host=host, port=port, database=dbname)
        self.engine = create_engine(self.connection_url)

    @contextmanager
    def create_connection(self):
        
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        """Open a transaction that commits on success, rolls back on failure."""
        with self.engine.begin() as conn:
            yield conn

    def load_query(self, sql, params=None, conn=None) -> pd.DataFrame:
        """Load data from db into a df in memory"""
        if conn is not None:
            return pd.read_sql_query(text(sql), conn, params=params)
        with self.create_connection() as local_conn:
            return pd.read_sql_query(text(sql), local_conn, params=params)

    def execute(self, query: str, params) -> int:
        sql = text(query)
        with self.create_connection() as conn:
            result = conn.execute(sql, params or ())
            conn.commit()
            return result.row_count

    def execute_script(self, sql_script: str, params=None, conn=None):
        """Execute a multi-statement SQL script sequentially."""
        statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]

        def _run(target_conn):
            """Executes a multi-statement SQL script sequentially."""
            for stmt in statements:
                target_conn.execute(text(stmt), params or {})

        if conn is not None:
            _run(conn)
            return
        with self.create_connection() as local_conn:
            _run(local_conn)
            local_conn.commit()
    
    def pandas_to_sql(self, df: pd.DataFrame, table_name, target_conn=None):
        if target_conn is not None:
            df.to_sql(name=table_name, con=target_conn, if_exists='append', index=False)
            return None
        
        with self.create_connection() as conn:
            df.to_sql(name=table_name, con=conn, if_exists='append', index=False)
            conn.commit()

def create_datamanager():
    return Database(
        dbname=os.getenv('DBNAME'),
        dbuser=os.getenv('DBUSER'),
        dbpassword=os.getenv('DBPASSWORD'),
    )

if __name__ == '__main__':
    from testing import CREATE_BOOKINGS_TABLE_SQL
    datamanager = create_datamanager()
    datamanager.pandas_to_sql()