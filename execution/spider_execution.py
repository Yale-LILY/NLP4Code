import sqlite3
import pandas as pd
import numpy as np
import re
import keyword

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
        print(f"Error {str(e)} in execution sql query {sql}")
        return None

def db_to_df_dict(conn: sqlite3.Connection) -> Dict[str, pd.DataFrame]:
    df_dict = {}
    for table_name in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        # modify to change everything including labels lower case
        df = pd.read_sql_query(f"SELECT * FROM {table_name[0]}", conn)
        df = df.applymap(lambda s: s.lower() if type(s) == str else s)
        df_dict[table_name[0].lower()] = df
        df_dict[table_name[0].lower()].rename(columns=lambda x: x.lower(), inplace=True)
    return df_dict

def spider_execution_py(code: str, df_dict: Dict[str, pd.DataFrame]) -> Any:
    local_vars = {"df_dict": df_dict}

    # use the tables as part of the code context
    table_vars_code = "import pandas as pd\n"
    for table_name in df_dict.keys():
        # table names may be reserved words like "class"
        if table_name in keyword.kwlist:
            table_vars_code += f"_{table_name} = df_dict['{table_name}']\n"
            code = code.replace(table_name, f"_{table_name}")
        else:
            table_vars_code += f"{table_name} = df_dict['{table_name}']\n"

    # lower everything in quotes
    code = re.sub(r"'(.*?)'", lambda p: f"'{p.group(1).lower()}'", code)
    # move select statements after sorting or drop_dup
    # TODO further processing needed
    code = re.sub(r"(.*?)(\[\[?.*?\]?\])(\.sort_values.*)", r"\1\3\2", code)
    code = table_vars_code + "\n" + f"answer = {code}"

    # execute the code
    try:
        exec_result = exec(code, {}, local_vars)

        if "answer" in local_vars:
            return local_vars["answer"]
        else:
            return None
    except Exception as e:
        print(f"error {str(e)} in execution code {code}")
        return None

def flatten_list_of_list(l: List[List[Any]]) -> List[Any]:
    result = []
    for sublist in l:
        if isinstance(sublist, list) or isinstance(sublist, tuple):
            result.extend(sublist)
        else:
            result.append(sublist)

    return result

def list_to_lower_case(l: List[Any]):
    result = []
    for object in l:
        if isinstance(object, str):
            result.append(object.lower())
        else:
            result.append(object)
    return result

def spider_answer_eq(prediction: Union[pd.DataFrame, pd.Series, List[Tuple[Any]]],
                     gold_answer: Union[List[Tuple[Any]], int]) -> bool:

    if isinstance(prediction, int) or isinstance(prediction, float):
        prediction = [prediction]

    if isinstance(prediction, list) or isinstance(prediction, np.ndarray):
        if isinstance(gold_answer, list):
            gold_flattened = list_to_lower_case(
                flatten_list_of_list(gold_answer))
            pred_flattened = flatten_list_of_list(prediction)
            result = pred_flattened == gold_flattened
        else:
            result = False
    elif isinstance(prediction, pd.DataFrame):
        if isinstance(gold_answer, list):
            # convert the dataframe to a list of tuples and check
            pred_list = flatten_list_of_list(list(prediction.itertuples(index=False, name=None)))
            gold_list = list_to_lower_case(flatten_list_of_list(gold_answer))
            result = pred_list == gold_list
        else:
            result = False
    elif isinstance(prediction, pd.Series):
        if isinstance(gold_answer, list):
            # convert the series to a list of tuples and check
            pred_list = flatten_list_of_list(prediction.tolist())
            gold_list = list_to_lower_case(flatten_list_of_list(gold_answer))
            result = pred_list == gold_list
        else:
            result = False
    else:
        # raise ValueError("prediction must be a pandas dataframe or series, but is a {}".format(type(prediction)))
        result = False

    return result