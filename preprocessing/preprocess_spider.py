import json
import sqlite3

from tqdm import tqdm
from typing import Dict, List, Any

from execution.spider_execution import connect_databse, spider_execution_sql

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

def preprocess_dataset(file_path: str, output_path: str) -> None:
    data = load_json(file_path)

    result = []
    for example in tqdm(data):
        db_file_path = f"data/spider/database/{example['db_id']}/{example['db_id']}.sqlite"
        conn = connect_databse(db_file_path)

        add_db_header_info(example, conn)
        
        if add_answer_info(example, conn):
            result.append(example)            

        conn.close()

    print(f"{len(result)} examples out of {len(data)} are processed.")

    with open(output_path, "w+") as f:
        for example in result:
            json.dump(example, f)
            f.write("\n")

def main():
    print(f"Preprocessing train data from {TRAIN_DATA_PATH}...")
    preprocess_dataset(TRAIN_DATA_PATH, "data/spider/train_spider_processed.jsonl")
    print(f"Preprocessing dev data from {DEV_DATA_PATH}...")
    preprocess_dataset(DEV_DATA_PATH, "data/spider/dev_processed.jsonl")

if __name__ == "__main__":
    main()