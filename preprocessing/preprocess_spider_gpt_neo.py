import json
import sqlite3

from tqdm import tqdm
from typing import Dict, List, Any

import os

TRAIN_DATA_PATH = "data/spider/train_spider.json"
DEV_DATA_PATH = "data/spider/dev.json"
DB_DATA_PATH = "data/spider/tables.json"
DB_DIR = "data/spider/database"

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


def generate_question_context_1(question: str, db_id: str, db: Dict[int, Any]) -> str:
    """
    What is Kyle’s id? | network_1 | highschooler : id, name ( Kyle ), grade | friend : student_id, friend_id | likes : student_id, liked_id
    """
    result = "{} | {}".format(question, db_id)
    for table_id in sorted(db.keys()):
        table_name = db[table_id]["table_name"]
        column_names = db[table_id]["column_names"]
        result += f" | {table_name} : "
        result += ", ".join(column_names)
    
    return result

def generate_question_context_2(question: str, db_id: str, db: Dict[int, Any]) -> str:
    """
    ### SQLite SQL tables, with their properties: 
    # 
    # Highschooler(ID, name, grade) 
    # Friend(student_id, friend_id) 
    # Likes(student_id, liked_id) 
    # 
    ### What is Kyle’s id?
    SELECT
    """
    result = "### SQLite SQL tables, with their properties:\n#\n"
    for table_id in sorted(db.keys()):
        table_name = db[table_id]["table_name"]
        column_names = db[table_id]["column_names"]
        result += f"# {table_name}("
        result += ", ".join(column_names) + ")\n"
    result += "#\n"
    result += f"### {question}\nSELECT"
    
    return result
        

def spider_execution_sql(sql: str, db_id: str, return_error_msg: bool = False) -> Any:
    
    db_path = os.path.join(DB_DIR, db_id, db_id + ".sqlite")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)

        return cursor.fetchall(), True
    except sqlite3.OperationalError as e:
        error_msg = f"ERROR: {str(e)}"
        print(f"Error {str(e)} in execution sql query {sql}")
        if return_error_msg:
            return error_msg, False
        else:
            return None, False

def preprocess_dataset(file_path: str, table_file_path: str, output_path: str, generate_question_context_func: callable) -> None:
    data = load_json(file_path)
    db_data = load_json(table_file_path)

    db_info = {}
    for db in tqdm(db_data):
        db_info[db["db_id"]] = {}
        for i, table_name in enumerate(db["table_names_original"]):
            db_info[db["db_id"]][i] = {}
            db_info[db["db_id"]][i]["table_name"] = table_name
            db_info[db["db_id"]][i]["column_names"] = [col[1] for col in db["column_names_original"] if col[0] == i]


    result = []
    for example in tqdm(data):
        question = example["question"]
        db_id = example["db_id"]
        db = db_info[db_id]
        question_context = generate_question_context_func(question, db_id, db)
        code = example["query"]

        answer, no_err = spider_execution_sql(code, db_id)
        if no_err:
            result.append({"text": question_context, "code": code, "answer": answer, "db_id": db_id})

    with open(output_path, "w") as f:
        f.write("\n".join([json.dumps(data) for data in result]))

    print(f"{len(result)} examples out of {len(data)} are processed.")
    return result


def main():
    print(f"Preprocessing train data from {TRAIN_DATA_PATH}...")
    preprocess_dataset(TRAIN_DATA_PATH, DB_DATA_PATH, "data/spider_processed/train_processed_2.jsonl", generate_question_context_2)
    print(f"Preprocessing dev data from {DEV_DATA_PATH}...")
    preprocess_dataset(DEV_DATA_PATH, DB_DATA_PATH, "data/spider_processed/dev_processed_2.jsonl", generate_question_context_2)

if __name__ == "__main__":
    main()