from multiprocessing.sharedctypes import Value
import sqlite3
import pandas as pd

from typing import List, Dict, Any, Union, Tuple

# from .safe_execution_util import execute

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

def db_to_df_dict(conn: sqlite3.Connection) -> Dict[str, pd.DataFrame]:
    df_dict = {}
    for table_name in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        df_dict[table_name[0]] = pd.read_sql_query(f"SELECT * FROM {table_name[0]}", conn)
        df_dict[table_name[0]].rename(columns=lambda x: x.lower(), inplace=True)
    return df_dict

def spider_execution_py(code: str, df_dict: Dict[str, pd.DataFrame]) -> Any:
    local_vars = {"df_dict": df_dict}

    # use the tables as part of the code context
    table_vars_code = "import pandas as pd\n"
    for table_name in df_dict.keys():
        table_vars_code += f"{table_name} = df_dict['{table_name}']\n"
    code = table_vars_code + "\n" + f"answer = {code}"

    # execute the code
    try:
        exec_result = exec(code, {}, local_vars)

        if "answer" in local_vars:
            return local_vars["answer"]
        else:
            return None
    except Exception as e:
        print(e)
        return None

def spider_answer_eq(prediction: Union[pd.DataFrame, pd.Series], 
                     gold_answer: Union[List[Tuple[Any]], int ,float]) -> bool:
    
    if isinstance(prediction, pd.DataFrame):
        if isinstance(gold_answer, list):
            # convert the dataframe to a list of tuples and check
            pred_list = list(prediction.itertuples(index=False, name=None))
            result = pred_list == gold_answer
        else:
            result = False
    elif isinstance(prediction, pd.Series):
        if isinstance(gold_answer, list):
            # convert the series to a list of tuples and check
            pred_list = [(x,) for x in prediction.tolist()]
            result = pred_list == gold_answer
        else:
            result = False
    else:
        raise ValueError("prediction must be a pandas dataframe or series, but is a {}".format(type(prediction)))

    return result