import json
import sqlite3
import pandas as pd

from tqdm import tqdm
from typing import Dict, List, Any

from execution.spider_execution import connect_databse, spider_execution_sql
from execution.spider_execution import db_to_df_dict, spider_execution_py, spider_answer_eq

from preprocessing.preprocess_spider import get_database_header


DATA_PATH = "data/squall/squall.json"
DB_DIR = "data/squall/tables/db"

COLUMN_DICT_FILE = "data/squall/column_dict.json"

"""
Squall's example looks like this (omit items that are not important):
    * nt: question ID
    * tbl: Table ID
    * columns: a list of processed table columns with the format of [raw header text, tokenized header text, 
        available column suffixes (ways to interpret this column beyond raw texts), column data type]
    * sql: tokenized SQL queries, each token has the format of [SQL type, value, span indices], SQL type is 
        one among Keyword, Column, Literal.Number, Literal.String. If the token is a literal, then the span 
        indices include the beginning and end indices to extract the literal from nl.

After processing, it should have the following format:
    * id: example ID
    * question: the nl question
    * sql: the sql format of the output
    * py: the python format of the output
    * answer: the answer of the query
    * metadata: the original Squall example
        * Note: the metadata should not contain things for learning
"""

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def process_squall_example(example: Dict[str, Any]) -> Dict[str, Any]:
    processed_example = {"question": " ".join(example["nl"]), 
                         "metadata": example}

    # get the db connection
    db_file_path = f"{DB_DIR}/{example['tbl']}.db"
    conn = connect_databse(db_file_path)
    df_dict: Dict[str, pd.DataFrame] = db_to_df_dict(conn)
    if len(df_dict) != 1:
        print(f"{len(df_dict)} tables found in {db_file_path}")

    
    # convert the column names to the original ones in the df_dict
    df_dict_headers = {}
    for table_name, df in df_dict.items():
        df_dict[table_name].rename(columns=lambda x: original_col_name_dict[x], inplace=True)
        df_dict_headers[table_name] = list(df.columns)
    processed_example["db_table_headers"] = df_dict_headers

    # get the sql query
    processed_sql_tokens = []
    for token_fields in example["sql"]:
        if token_fields[0] == "Column":
            processed_sql_tokens.append(original_col_name_dict[token_fields[1]])
        else:
            processed_sql_tokens.append(token_fields[1])
    sql_query = " ".join(processed_sql_tokens)
    processed_example["sql"] = sql_query

    # verify the sql query
    sql_exec_result = spider_execution_sql(sql_query, conn) 

    print("")

    pass

def build_column_name_dict(dataset: List[Dict[str, Any]]):
    db_column_dict = {}

    # build the dict to convert back the original column names
    for example in tqdm(dataset):
        table_id = example["tbl"]

        if table_id in db_column_dict:
            continue

        original_col_name_dict = {"id": "id", "agg": "agg"} # these two will stay the same
        db_column_fields = example["columns"]
        for i, field in enumerate(db_column_fields):
            original_col_name_dict[f"c{i+1}"] = "_".join(field[1])
            for suffix in field[2]:
                original_col_name_dict[f"c{i+1}_{suffix}"] = "_".join(field[1]) + "_" + suffix

        db_column_dict[example["tbl"]] = original_col_name_dict
    
    print(f"Built the dict for {len(db_column_dict)} tables")

    # dump to file
    with open(COLUMN_DICT_FILE, "w") as f:
        for table_id, original_col_name_dict in db_column_dict.items():
            f.write(f"{table_id}: {original_col_name_dict}\n")

def preprocess_squall_dataset(dataset: List[Dict[str, Any]]):

    processed_data = []
    for example in tqdm(dataset):
        processed_data.append(process_squall_example(example))


def main():
    # read the data
    data = load_json(DATA_PATH)
    # preprocess_squall_dataset(data)

    build_column_name_dict(data)

if __name__ == "__main__":
    main()