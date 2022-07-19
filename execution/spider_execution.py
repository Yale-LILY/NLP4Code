import sqlite3
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Union, Tuple

# from .safe_execution_util import execute
DB_DIR = "data/spider/database"


def connect_databse(db_path: str) -> sqlite3.Connection:
    # connect the database with read-only access
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    return conn

def spider_execution_sql(sql: str, example: Dict[str, Any], return_error_msg: bool = False) -> Any:
    
    db_id = example["db_id"]
    db_path = os.path.join(DB_DIR, db_id, db_id + ".sqlite")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        return cursor.fetchall()
    # except sqlite3.OperationalError as e:
    #     error_msg = f"ERROR: {str(e)}"
    #     print(f"Error {str(e)} in execution sql query {sql}")
    #     if return_error_msg:
    #         return error_msg
    #     else:
    #         return None
    except:
        return None

def db_to_df_dict(conn: sqlite3.Connection) -> Dict[str, pd.DataFrame]:
    df_dict = {}
    for table_name in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        df_dict[table_name[0]] = pd.read_sql_query(f"SELECT * FROM {table_name[0]}", conn)
        df_dict[table_name[0]].rename(columns=lambda x: x.lower(), inplace=True)
    return df_dict

def spider_execution_py(code: str, df_dict: Dict[str, pd.DataFrame], return_error_msg: bool = False) -> Any:
    local_vars = {"df_dict": df_dict}

    # use the tables as part of the code context
    table_vars_code = "import pandas as pd\n"
    for table_name in df_dict.keys():
        table_vars_code += f"# {' '.join(list(df_dict[table_name].columns))}\n{table_name} = df_dict['{table_name}']\n"
    code = table_vars_code + "\n" + code

    # execute the code
    try:
        exec_result = exec(code, {}, local_vars)

        if "answer" in local_vars:
            return local_vars["answer"]
        else:
            return None
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(f"error {str(e)} in execution code {code}")
        if return_error_msg:
            return error_msg
        else:
            return None

def flatten_list_of_list(l: List[List[Any]]) -> List[Any]:
    result = []
    for sublist in l:
        if isinstance(sublist, list) or isinstance(sublist, tuple):
            result.extend(sublist)
        else:
            result.append(sublist)

    return result

def spider_answer_eq(prediction: Union[pd.DataFrame, pd.Series, List[Tuple[Any]]], 
                     gold_answer: Union[List[Tuple[Any]], int]) -> bool:

    if isinstance(prediction, int) or isinstance(prediction, float):
        prediction = [prediction]
    
    if isinstance(prediction, list) or isinstance(prediction, np.ndarray):
        if isinstance(gold_answer, list):
            gold_flattened = flatten_list_of_list(gold_answer)
            pred_flattened = flatten_list_of_list(prediction)
            result = pred_flattened == gold_flattened
        else:
            result = False
    elif isinstance(prediction, pd.DataFrame):
        if isinstance(gold_answer, list):
            # convert the dataframe to a list of tuples and check
            pred_list = flatten_list_of_list(list(prediction.itertuples(index=False, name=None)))
            gold_list = flatten_list_of_list(gold_answer)
            result = pred_list == gold_list
        else:
            result = False
    elif isinstance(prediction, pd.Series):
        if isinstance(gold_answer, list):
            # convert the series to a list of tuples and check
            pred_list = flatten_list_of_list(prediction.tolist())
            gold_list = flatten_list_of_list(gold_answer)
            result = pred_list == gold_list 
        else:
            result = False
    else:
        # raise ValueError("prediction must be a pandas dataframe or series, but is a {}".format(type(prediction)))
        result = False

    return result