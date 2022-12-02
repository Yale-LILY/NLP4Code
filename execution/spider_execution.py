import sqlite3
import json
import pandas as pd
import numpy as np
import os
import shutil
import string
import random
import re
import multiprocessing
import keyword
import math

from typing import List, Dict, Any, Union, Tuple
from pathlib import Path

from execution.wtq_eval import wtq_answer_eq

from execution.spider_official_exec_match import eval_exec_match

# from .safe_execution_util import execute

def pd_df_to_dict(df: pd.DataFrame) -> Tuple[dict, bool]:
    cutoff = False
    if len(df) > 5:
        df = df.head(5)
        cutoff = True
    return df.to_dict(orient='tight'), cutoff

def pd_df_from_dict(dt: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(dt, orient='tight')

def direct_categorize(example: Dict[str, Any]) -> List[str]:
    return [example["category"]]

def spider_categorize_complexity(example: Dict[str, Any]) -> List[str]:
    sql = example["query"]

    tags = []
    if 'join' in sql.lower():
        tags.append("JOIN")
    if any([s.lower() in sql.lower() for s in ["INTERSECT", "UNION", "EXCEPT"]]):
        tags.append("COMPOUND")
    if '(select' in sql.lower() or '( select' in sql.lower():
        tags.append("NESTED")
    
    if len(tags) == 0:
        tags.append("SIMPLE")
    
    return tags

def connect_databse(db_path: str, read_only: bool=True) -> sqlite3.Connection:
    # connect the database with read-only access
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
    return conn

def step_wise_thread_execution_sql(sql: str, conn: sqlite3.Connection = None) -> Tuple[pd.DataFrame, str]:
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    def worker(sql: str, conn: sqlite3.Connection, return_dict: Dict[str, Any]):
        state, error_msg = step_wise_execution_sql(sql, conn)
        return_dict["state"] = state
        return_dict["error_msg"] = error_msg

    p = multiprocessing.Process(target=worker, args=(sql, conn, return_dict))
    p.start()
    p.join(timeout=10)

    if len(return_dict) == 0:
        print(f"Timeout for sql: {sql}")
        return None, "TIMEOUT"
    else:
        return return_dict["state"], return_dict["error_msg"]

def step_wise_execution_sql(sql: str, conn: sqlite3.Connection) -> Tuple[pd.DataFrame, str]:
    # sql = "SELECT " + sql
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        tmp_tab_var = sql.split(' AS ')[0].split(' ')[-1]
        state = pd.read_sql_query(f"SELECT * FROM {tmp_tab_var}", conn)

        error_msg = None
    except Exception as e:
        state = None
        error_msg = f"ERROR: {str(e)}"
        # print(f"Error {str(e)} in execution sql query {sql}")

    return state, error_msg

def spider_decomp_execution_sql(sql: str, example: Dict[str, Any], return_error_msg: bool = False) -> bool:
    # preprocess to lower and change quotes
    sql = sql.replace("\"", "'")
    sql = re.sub(r"\b(?<!')(\w+)(?!')\b", lambda match: match.group(1).lower(), sql) 

    decomp_sqls = sql.split(" || ")
    if not all([s.lower().startswith("create table tmp_tab_") for s in decomp_sqls]):
        return None

    # sequentially execute the sqls
    db_path = example["db_path"]
    extension = db_path.split(".")[-1]
    db_path_root = Path(db_path).parent.parent.absolute()

    # copy file to be safe
    random_file_name = int(abs(hash(db_path+example["query"]+example["question"]))) % 10000000
    tmp_db_path = os.path.join(str(db_path_root), 'tmp', f"tmp_{random_file_name}.{extension}")
    shutil.copyfile(db_path, tmp_db_path) 
    conn = sqlite3.connect(f"file:{tmp_db_path}?mode=rw", uri=True)
    cursor = conn.cursor()

    for decomp_sql in decomp_sqls:
        try:
            cursor.execute(decomp_sql)
            error_msg = None
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            # print(f"Error {str(e)} in execution sql query {sql}")
    
    # cleanup
    conn.commit()
    conn.close()
    
    last_tmp_table_number = 'tmp_tab_' + decomp_sqls[-1].split('tmp_tab_')[1].split(' ')[0]
    final_sql = f"SELECT * FROM {last_tmp_table_number}"

    if error_msg is None:
        # sql = "SELECT " + sql
        result = bool(eval_exec_match(tmp_db_path, db_path, final_sql, example["query"], plug_value=False, keep_distinct=False, progress_bar_for_each_datapoint=False))
    else:
        result = None

    os.remove(tmp_db_path)
    return result

def post_process_wtq_exec_result(sql: str, exec_result: Any, metadata: Dict[str, Any]) -> Any:
    if len(exec_result) == 0 or exec_result is None or exec_result[0] is None or exec_result[0][0] is None:
        return exec_result

    if exec_result[0][0] in [0, 1]:
        if "above or below" in metadata["question"] or "below or above" in metadata["question"]:
            return [[["below", "above"][int(exec_result[0][0])]]]
        elif "more or less" in metadata["question"]:
            return [[["less", "more"][int(exec_result[0][0])]]]
        elif "before or after" in metadata["question"]:
            return [[["after", "before"][int(exec_result[0][0])]]]
        elif metadata["question"].split(" ")[0] in ["is", "are", "does", "do", "was", "were"]:
            return [[["no", "yes"][int(exec_result[0][0])]]]
        else:
            return exec_result
    elif "which month" in metadata["question"] and exec_result[0][0] in list(range(1, 13)):
        return [[["January", "February", "March", "April", "May", "June", "July", \
                  "August", "September", "October", "November", "December"][int(exec_result[0][0]) - 1]]]
    # elif is_number(exec_result[0][0]) and float(exec_result[0][0]) > 3100 and float(exec_result[0][0]).is_integer():
    #     return [[format(int(exec_result[0][0]), ",")]]
    else:
        return exec_result

def squall_official_execution_sql(sql: str, example: Dict[str, Any], return_error_msg: bool = False) -> bool:
    exec_result = spider_execution_pd_sql(sql, example)
    if exec_result is not None:
        list_exec_result = exec_result.values.tolist()
        list_exec_result = post_process_wtq_exec_result(sql, list_exec_result, example)
        return wtq_answer_eq(list_exec_result, example["original_answer"])
    else:
        return None

def squall_execution_sql(sql: str, example: Dict[str, Any], return_error_msg: bool = False) -> bool:
    db_path = example["db_path"]

    # to evaluate executability
    exec_result = spider_execution_sql(sql, example)
    if exec_result is not None:
        if squall_answer_eq(exec_result, example["original_answer"]):
            return True
        else:
            print(f"Execution result {exec_result} does not match with gold answer {example['original_answer']}")
        return bool(eval_exec_match(db_path, db_path, sql, example["query"], plug_value=False, keep_distinct=False, progress_bar_for_each_datapoint=False))
    else:
        return None

def spider_official_execution_sql(sql: str, example: Dict[str, Any], return_error_msg: bool = False, keep_distinct: bool = False) -> bool:
    db_path = example["db_path"]

    # to evaluate executability
    exec_result = spider_execution_sql(sql, example)
    if exec_result is not None:
        # sql = "SELECT " + sql
        return bool(eval_exec_match(db_path, db_path, sql, example["query"], plug_value=False, keep_distinct=keep_distinct, progress_bar_for_each_datapoint=False))
    else:
        return None

def spider_official_answer_eq(prediction: Union[pd.DataFrame, pd.Series, List[Tuple[Any]]], 
                     gold_answer: Union[List[Tuple[Any]], int]) -> bool:
    return prediction

def spider_execution_pd_sql(sql: str, example: Dict[str, Any], conn: sqlite3.Connection = None) -> pd.DataFrame:

    if conn is None:
        db_path = example["db_path"]
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        result = pd.read_sql_query(sql, conn)
    except Exception as e:
        result = None
    
    return result

def spider_execution_sql(sql: str, example: Dict[str, Any], return_error_msg: bool = False) -> Any:
    db_path = example["db_path"]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        return cursor.fetchall()
    except:
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

def spider_execution_py(code: str, df_dict: Dict[str, pd.DataFrame], return_error_msg: bool = False) -> Any:
    # copy the dataframes to avoid functional side effects
    copied_df_dict = dict([(k, v.copy(deep=True)) for k, v in df_dict.items()])
    df_dict = copied_df_dict

    local_vars = {"df_dict": df_dict}

    # use the tables as part of the code context
    table_vars_code = "import pandas as pd\n"
    for table_name in df_dict.keys():
        # table names may be reserved words like "class"
        if table_name in keyword.kwlist:
            table_vars_code += f"_{table_name} = df_dict['{table_name}']\n"
            # but we have to make sure that table columns are not changed
            # code = code.replace(table_name, f"_{table_name}")
            code = re.sub("((?<!_)class(?!_))", "_class", code)
        else:
            table_vars_code += f"{table_name} = df_dict['{table_name}']\n"

    # lower everything in quotes
    code = re.sub(r"'(.*?)'", lambda p: f"'{p.group(1).lower()}'", code)
    # move select statements after sorting or drop_dup
    # TODO further processing needed, case 784, 1721,
    #  and select, drop_duplicate, followed by sorting
    code = re.sub(r"(.*(?<!\[))(\[\[?.*?\]?\])(\.sort_values.*)", r"\1\3\2", code)
    code = table_vars_code + "\n" + f"answer = {code}"

    # execute the code
    try:
        exec_result = exec(code, {}, local_vars)

        if "answer" in local_vars:
            return local_vars["answer"]
        else:
            return None
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(f"\033[1m RUNNING... \033[0m")
        print(f"error {str(e)} in execution code {code}")
        if return_error_msg:
            return error_msg
        else:
            return None

def flatten_list_of_list(l: List[List[Any]], sort: bool = False) -> List[Any]:
    result = []
    for sublist in l:
        if isinstance(sublist, list) or isinstance(sublist, tuple):
            result.extend(sublist)
        else:
            result.append(sublist)

    if sort:
        result.sort(key = str)
        return result
    else:
        return result

def list_to_lower_case(l: List[Any]):
    result = []
    for object in l:
        if isinstance(object, str):
            result.append(object.lower())
        else:
            result.append(object)
    return result

def compare_lists(l1: List[Any], l2: List[Any]) -> bool:
    if len(l1) != len(l2):
        return False
    else:
        for i in range(len(l1)):
            if type(l1[i]) == float:
                if not math.isclose(l1[i], l2[i]):
                    return False
                else:
                    continue
            elif l1[i] != l2[i]:
                return False
        return True

def squall_answer_eq(prediction: Any, gold_answer: Any) -> bool:
    assert isinstance(gold_answer, str)
    if prediction is None:
        return False
    if len(prediction) == 0:
        return False
    
    if not len(prediction) == 1 and len(prediction[0]) == 1: 
        # print(f"prediction is {prediction}, and gold_answer is {gold_answer}")
        return False

    prediction = prediction[0][0]
    if isinstance(prediction, str):
        result = prediction == gold_answer.lower()
    elif isinstance(prediction, int) or isinstance(prediction, float):
        try:
            float_answer = float(gold_answer)
            result = float_answer == float(prediction)
        except ValueError:
            if gold_answer.split(' ')[0].isnumeric():
                result = float(gold_answer.split(' ')[0]) == float(prediction)
            else:
                result = False
    elif prediction is None:
        result = False
    else:
        raise ValueError(f"Unsupported type {type(prediction)}")
                
    return result

def spider_answer_eq(prediction: Union[pd.DataFrame, pd.Series, List[Tuple[Any]]],
                     gold_answer: Union[List    [Tuple[Any]], int],
                     sort: bool = False) -> bool:

    try:
        if isinstance(prediction, int) or isinstance(prediction, float) or (not isinstance(prediction, list) and not isinstance(prediction, pd.DataFrame) and not isinstance(prediction, np.ndarray) and not isinstance(prediction, tuple) and np.issubdtype(prediction, np.integer)):
            prediction = [prediction]

        if isinstance(prediction, list) or isinstance(prediction, np.ndarray):
            if isinstance(gold_answer, list):
                gold_flattened = list_to_lower_case(
                    flatten_list_of_list(gold_answer, sort))
                pred_flattened = flatten_list_of_list(prediction, sort)
                result = compare_lists(pred_flattened, gold_flattened)
            else:
                result = False
        elif isinstance(prediction, pd.DataFrame):
            if isinstance(gold_answer, list):
                # we include the index only when it exists
                pred_list = flatten_list_of_list(list(prediction.itertuples(
                    index=bool(prediction.index.name), name=None)), sort)
                gold_list = list_to_lower_case(flatten_list_of_list(gold_answer, sort))
                result = compare_lists(pred_list, gold_list)
            else:
                result = False
        elif isinstance(prediction, pd.Series):
            if isinstance(gold_answer, list):
                # convert the series to a list of tuples and check
                # we include the index only when it exists
                if prediction.index.name:
                    pred_list = flatten_list_of_list(list(prediction.items()), sort)
                else:
                    pred_list = flatten_list_of_list(prediction.tolist(), sort)
                gold_list = list_to_lower_case(flatten_list_of_list(gold_answer, sort))
                result = compare_lists(pred_list, gold_list)
            else:
                result = False
        else:
            # raise ValueError("prediction must be a pandas dataframe or series, but is a {}".format(type(prediction)))
            result = False

        return result
    except Exception as e:
        print("##############################")
        print(f"Error in spider_answer_eq: {repr(e)}")
        print("##############################")
        return False

if __name__ == "__main__":
    with open("data/squall/squall_processed_dev_all.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]

    # f1 = open("squall_process.log", "w+")

    # verifiy that the evaluation is correct
    eval_correct = 0
    total = 0
    for example in train_data:
        sql = example["query"]
        exec_result = squall_official_execution_sql(sql, example)

        if exec_result:
            eval_correct += 1
        else:
            if exec_result is not None:
                print(f"question: {example['question']}")
                print(f"sql: {sql}")
                print(f"original_answer: {example['original_answer']}, but got: {exec_result}")
                # f1.write(f"original_answer: {example['original_answer']}, but got: {exec_result}")
        total += 1

        # f1.flush()
        print(f"number of correct: {eval_correct} / {total}, of which {eval_correct * 100 / total}%")
    
    # f1.close()