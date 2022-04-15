import json
import sqlite3
import random

random.seed(333)

from tqdm import tqdm
from typing import Dict, List, Any

from execution.spider_execution import connect_databse, spider_execution_sql
from execution.spider_execution import db_to_df_dict, spider_execution_py, spider_answer_eq

from parsing.preprocess import sql_query_to_pandas_code_snippets


TRAIN_DATA_PATH = "data/spider/train_spider.json"
DEV_DATA_PATH = "data/spider/dev.json"

"""
Spider's example looks like this:
    "db_id": "department_management",
    "query": "SELECT count(*) FROM head WHERE age  >  56",
    "question": "How many heads of the departments are older than 56 ?",
"""

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def get_database_header(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")

    result = {}
    for row in cursor.fetchall():
        table_name = row[1]
        cursor.execute(f"SELECT * FROM {table_name}")
        headers = [desc[0] for desc in cursor.description]
        result[table_name] = headers

    return result 

def add_db_header_info(example: Dict[str, Any], conn: sqlite3.Connection) -> None:
    # add the headers of the associated databases to the example
    headers = get_database_header(conn)

    example["db_table_headers"] = headers

def add_answer_info(example: Dict[str, Any], conn: sqlite3.Connection) -> None:
    # add the answer to the example
    exec_result = spider_execution_sql(example["query"], conn)

    if exec_result is not None:
        example["answer"] = exec_result
        return True
    else:
        return False

def verify_py_code(example: Dict[str, Any], conn: sqlite3.Connection) -> bool:
    assert "pandas_converted" in example, "pandas_converted not in example"

    py_code = example["pandas_converted"]

    df_dict = db_to_df_dict(conn)

    exec_result = spider_execution_py(py_code, df_dict)

    if exec_result is not None:
        result = spider_answer_eq(exec_result, example["answer"])
        return result
    else:
        return False

def convert_and_preprocess_dataset(file_path: str, output_path: str) -> None:
    data = load_json(file_path)
    random.shuffle(data)
    data = data[:100]


    result = []
    for example in tqdm(data):
        # first get the database in two forms
        db_file_path = f"data/spider/database/{example['db_id']}/{example['db_id']}.sqlite"
        conn = connect_databse(db_file_path)
        df_dict = db_to_df_dict(conn)

        # perform the conversion and verify the result
        converted_py_code = "\n".join(sql_query_to_pandas_code_snippets(example["query"].lower()))
        sql_exec_result = spider_execution_sql(example["query"], conn)
        py_exec_result = spider_execution_py(converted_py_code, df_dict)

        if py_exec_result is not None:
            match_result = spider_answer_eq(py_exec_result, sql_exec_result)
        else:
            match_result = False
        
        # add the conversion result
        example["conversion"] = {
            "pandas_converted": converted_py_code,
            "py_exec_result": py_exec_result,
            "sql_exec_result": sql_exec_result,
            "answer_match": match_result
        }

        # other preprocessing
        add_db_header_info(example, conn)

        result.append(example)
        conn.close()

    print(f"{len(list(filter(lambda x: x['conversion']['answer_match'] is True, result)))} examples out of {len(result)} are successfully converted.")

    print("")
    with open(output_path, "w+") as f:
        for example in result:
            json.dump(example, f)
            f.write("\n")

def main():
    print(f"Preprocessing train data from {TRAIN_DATA_PATH}...")
    convert_and_preprocess_dataset(TRAIN_DATA_PATH, "data/spider/tmp.jsonl")
    # print(f"Preprocessing dev data from {DEV_DATA_PATH}...")
    # preprocess_dataset(DEV_DATA_PATH, "data/spider/dev_processed.jsonl")

if __name__ == "__main__":
    main()