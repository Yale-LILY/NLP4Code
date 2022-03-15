import sqlite3
import pandas as pd

from typing import List, Dict, Any

from .safe_execution_util import execute

def connect_databse(db_path: str) -> sqlite3.Connection:
    # connect the database with read-only access
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    return conn

def spider_execution_sql(sql: str, conn: sqlite3.Connection) -> Any:
    cursor = conn.cursor()

    try:
        cursor.execute(sql)

        return cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(e)
        return None

def spider_execution_py(code: str, df_dict: Dict[str, pd.DataFrame]) -> Any:
    pass

def spider_answer_eq(prediction: Any, gold_answer: Any) -> bool:
    pass