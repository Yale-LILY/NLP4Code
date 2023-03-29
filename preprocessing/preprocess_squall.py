import json
import sqlite3
from tkinter import W
import pandas as pd
import random
import os

from tqdm import tqdm
from typing import Dict, List, Any, Tuple

from execution.spider_execution import connect_databse, spider_execution_sql, squall_answer_eq
from execution.spider_execution import db_to_df_dict, spider_execution_py, spider_answer_eq, flatten_list_of_list
from execution.wtq_eval import wtq_answer_eq

from preprocessing.preprocess_spider import process_spider_output_example, pd_df_from_dict, post_process_exec_result
from finetuning.lightning_modules.datasets.spider_reader import example_to_demonstration_sql_bridge


DATA_PATH = "data/squall/wtq-test-w_answer.json"
# DATA_PATH = "data/squall/squall.json"
DB_DIR = "data/squall/tables/db"

COLUMN_DICT_FILE = "data/squall/column_mapping_dict_test.jsonl"
# COLUMN_DICT_FILE = "data/squall/column_mapping_dict.jsonl"
# COLUMN_DICT_FILE = "data/squall/column_dict.json"

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

def process_squall_example(example: Dict[str, Any], col_name_mapping: Dict[str, Dict[str, str]], add_sql: bool = True) -> Dict[str, Any]:
    processed_example = {"db_id": example["tbl"], "question": " ".join(example["nl"]), 
                         "query": "", "answer": "", "original_answer": example['tgt'],
                         "metadata": example}

    # get the db connection
    db_file_path = f"{DB_DIR}/{example['tbl']}.db"
    conn = connect_databse(db_file_path, read_only=False)
    df_dict: Dict[str, pd.DataFrame] = db_to_df_dict(conn)

    # get the original column dict
    original_col_name_dict = col_name_mapping[example["tbl"]]
    
    # add the table header information
    df_dict_headers = {}
    for table_name, df in df_dict.items():
        df_dict[table_name].rename(columns=lambda x: original_col_name_dict[x] if x in original_col_name_dict.keys() else x,  inplace=True)
        df_dict_headers[table_name] = list(df.columns)
    processed_example["db_table_headers"] = df_dict_headers

    # get the sql query
    if add_sql:
        processed_sql_tokens = []
        for token_fields in example["sql"]:
            if token_fields[0] == "Column" and token_fields[1] in original_col_name_dict: # to handle a edge case
                processed_sql_tokens.append(original_col_name_dict[token_fields[1]])
            else:
                processed_sql_tokens.append(token_fields[1])
        sql_query = " ".join(processed_sql_tokens)

        sql_query = name_change(sql_query, original_col_name_dict)
        sql_query = sql_query.replace(" from w", " from main_table")
        processed_example["query"] = sql_query

        # verify the sql query
        sql_exec_result = spider_execution_sql(sql_query, conn) 
        processed_example["answer"] = sql_exec_result

    return processed_example

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
        f.write(json.dumps(db_column_dict))
    #     for table_id, original_col_name_dict in db_column_dict.items():
    #         entry = {table_id: original_col_name_dict}
    #         f.write(json.dumps(entry)+"\n")

def post_process_column_mapping_dict(mapping_dict: Dict[str, str]):
    result_dict = {}
    for canonical_name, col_name in mapping_dict.items():
        if len(col_name) == 0:
            new_name = 'no_name'
        else:
            new_name = col_name
        new_name = new_name.replace('#', 'count')

        # replace all special tokens to underlines
        tmp_name = ''
        for c in new_name:
            if not (c.isalnum() or c == '_'):
                tmp_name += '_'
            else:
                tmp_name += c
        new_name = '_'.join(filter(lambda x: len(x) > 0, tmp_name.split('_')))

        if new_name != col_name:
            print(f"changed name from {col_name} to {new_name}")

        result_dict[canonical_name] = new_name
    
    # renaming duplicate column names
    value_set = set()
    for k, v in result_dict.items():
        if v in value_set:
            i = 2
            while f'{v}_{i}' in value_set:
                i += 1
            result_dict[k] = f'{v}_{i}'
            value_set.add(f'{v}_{i}')
        else:
            value_set.add(v)
    
    return result_dict

def name_change(old_name: str, col_naming_dict: Dict[str, str]) -> str:
    new_name = old_name
    
    # get the items and sort by key length to avoid double swap
    items = sorted(col_naming_dict.items(), key=lambda x: len(x[0]), reverse=True)
    for k, v in items:
        if len(v) > 0 and v[0].isnumeric(): # to handle numeric prefix column names
            v = f"\"{v}\""
        new_name = new_name.replace(k, v)

    return new_name

def revert_db_column_names():
    # get the canonical -> orignal name dicts
    original_col_name_dicts = json.load(open(COLUMN_DICT_FILE))
    items = [(k, v) for k, v in original_col_name_dicts.items()] # debug setting
    # random.shuffle(items)

    # save the new mapping dict into a file
    f = open('data/squall/column_mapping_dict.jsonl', 'w+')

    for db_name, col_naming_dict in tqdm(items):
        # post-process the col renaming dict
        col_naming_dict = post_process_column_mapping_dict(col_naming_dict)
        f.write(json.dumps({"db_id": db_name, "col_renaming_mapping": col_naming_dict})+"\n")

        # connec the db
        db_file_path = f"{DB_DIR}/{db_name}.db"
        conn = connect_databse(db_file_path, read_only=False)

        # get the pandas df version for easier access
        df_dict: Dict[str, pd.DataFrame] = db_to_df_dict(conn)

        # process each table
        for table_name in df_dict.keys():
            # first change the table name to semantic ones
            if table_name == 'w':
                new_table_name = 'main_table'
            else:
                new_table_name = name_change(table_name, col_naming_dict)
            if table_name != new_table_name:
                conn.execute(f"ALTER TABLE {table_name} RENAME TO {new_table_name};")

            # then change each one of the columns in the table
            for col_name in df_dict[table_name].columns:
                if col_name in col_naming_dict:
                    new_col_name = col_naming_dict[col_name]
                else:
                    new_col_name = col_name
                if new_col_name != col_name:
                    conn.execute(f"ALTER TABLE {new_table_name} RENAME COLUMN {col_name} TO \"{new_col_name}\";")
        
        conn.commit()
        df_dict: Dict[str, pd.DataFrame] = db_to_df_dict(conn)
        conn.close()


def preprocess_squall_dataset():
    # load the dataset
    dataset = load_json(DATA_PATH)

    # load the column renaming dict
    with open(COLUMN_DICT_FILE, 'r') as f:
        dict_items = [json.loads(s) for s in f.readlines()]
        # col_renaming_dict = {item['db_id']: item['col_renaming_mapping'] for item in dict_items}
        col_renaming_dict = dict_items[0]

    processed_data = []
    for example in tqdm(dataset):
        processed_data.append(process_squall_example(example, col_renaming_dict, add_sql=False))
        # executable_sql_n = sum(map(lambda x: float(x['answer'] is not None), processed_data))
        # print(f"{len(processed_data)} examples, {executable_sql_n} executable sql")
    
    with open("data/squall/squall_processed_test.jsonl", "w+") as f:
        for example in processed_data:
            f.write(json.dumps(example)+"\n")


def fix_squall_eval_with_original_answer(example: Dict[str, Any]):
    for program_dict in example["generated_programs"]:
        if isinstance(program_dict["exec_result"], str):
            assert program_dict["exec_result"] == "ERROR"
            continue
        # if program_dict["exec_match"]:
        #     continue
        exec_result = pd_df_from_dict(program_dict["exec_result"])
        list_exec_result = exec_result.values.tolist()
        list_exec_result = post_process_exec_result(list_exec_result, example["metadata"])
        if wtq_answer_eq(list_exec_result, example["metadata"]["original_answer"]):
            program_dict["exec_match"] = True
        else:
            # if program_dict['exec_match']:
            #     print(f"original answer {example['metadata']['original_answer']}, but exec result {list_exec_result}, " + \
            #         f" original exec result is {program_dict['exec_result']}")

            program_dict["exec_match"] = False
    
    return example

def train_dev_split():
    all_data_file_path = "data/squall/squall_processed.jsonl"

    with open(all_data_file_path, "r") as f:
        all_data = [json.loads(s) for s in f.readlines()]
    
    for example in all_data:
        example["db_path"] = os.path.join("data/squall/tables/db", f'{example["db_id"]}.db')

    dev_ids = []
    with open("data/squall/dev-4.ids", "r") as f:
        dev_ids = json.load(f)
    
    train_examples = list(filter(lambda x: x["db_id"] not in dev_ids, all_data))
    dev_examples = list(filter(lambda x: x["db_id"] in dev_ids, all_data))

    # save the train and dev examples
    with open("data/squall/squall_processed_train_all.jsonl", "w+") as f:
        for example in train_examples:
            f.write(json.dumps(example)+"\n")

    with open("data/squall/squall_processed_dev_all.jsonl", "w+") as f:
        for example in dev_examples:
            f.write(json.dumps(example)+"\n")


def verification_train_dev_split():
    all_data_file_path = "data/squall/codex_20_output_all.jsonl"

    with open(all_data_file_path, "r") as f:
        all_data = [json.loads(s) for s in f.readlines()]
    
    for i, example in enumerate(tqdm(all_data)):
        all_data[i] = fix_squall_eval_with_original_answer(example)
    
    pass_at_k = sum([any([y['exec_match'] for y in x['generated_programs']]) for x in all_data]) / len(all_data)
    print(f"upper bound is {pass_at_k}")

    dev_ids = []
    with open("data/squall/dev-4.ids", "r") as f:
        dev_ids = json.load(f)
    
    train_examples = list(filter(lambda x: x["metadata"]["db_id"] not in dev_ids, all_data))
    dev_examples = list(filter(lambda x: x["metadata"]["db_id"] in dev_ids, all_data))

    # save the train and dev examples
    with open("data/squall/codex_20_output_all_train.jsonl", "w+") as f:
        for example in train_examples:
            f.write(json.dumps(example)+"\n")

    with open("data/squall/codex_20_output_all_dev.jsonl", "w+") as f:
        for example in dev_examples:
            f.write(json.dumps(example)+"\n")

def all_data_aggregation():
    # get all the original wtq data
    all_output_files = [f"results_cot/squall_codex_davinci-few_shot_baseline-for-in_house_models-pass@20-{i}.jsonl" for i in ["left", "dev", "train"]]
    all_outputs = []
    for output_file in all_output_files:
        with open(output_file, "r") as f:
            all_outputs.extend([json.loads(s) for s in f.readlines()])

    # get all the processed data
    all_processed_files = [f"data/squall/codex_20_output_{i}_verification_db_path.jsonl" for i in ["dev", "train"]]
    all_processed_examples = []
    for processed_file in all_processed_files:
        with open(processed_file, "r") as f:
            all_processed_examples.extend([json.loads(s) for s in f.readlines()])
    db_id_question_set = {(example["metadata"]["db_id"], example["metadata"]["question"]) for example in all_processed_examples}

    # filte the original wtq data
    remaining_outputs = list(filter(lambda x: (x["example"]["db_id"], x["question"]) not in db_id_question_set, all_outputs))
    print(f"{len(all_processed_examples)} already processed")
    print(f"{len(remaining_outputs)} remaining outputs")

    for output in tqdm(remaining_outputs):
        metadata = output["example"]
        metadata["db_path"] = os.path.join("data/squall/tables/db", f'{metadata["db_id"]}.db')
        result_example_dict = {"metadata": metadata}

        processed_programs = []
        distinct_programs_counts = {}

        # store the information for the gold program
        gold_program = metadata["query"] 
        result_example_dict["gold_program"] = process_spider_output_example(gold_program, metadata)
        assert result_example_dict["gold_program"] is not None

        # add all the generated results
        # NOTE: the generated programs may be the same as the gold one, but we keep them separated during processing
        for code in output["all_output"][0]: 
            # process the raw code outputs
            processed_result = process_spider_output_example(code, metadata, distinct_programs_counts)

            if processed_result is not None:
                processed_programs.append(processed_result)
            else:
                continue
        
        # add the program count
        for processed_program in processed_programs:
            processed_program["program_count"] = distinct_programs_counts[processed_program["lower_code"]]
        
        result_example_dict["generated_programs"] = processed_programs
        all_processed_examples.append(result_example_dict)

    print(f"now total of {len(all_processed_examples)} processed")

    with open("data/squall/codex_20_output_all.jsonl", "w+") as f:
        for example in all_processed_examples:
            f.write(json.dumps(example)+"\n")

def get_example_val_for_db_columns(db_path: str, table_columns_dict: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    # return a dict of table name to a list of column name and example value pairs
    result_dict = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    for table_name, column_names in table_columns_dict.items():
        col_name_val_tuples = []
        for column_name in column_names:
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL"
            try:
                cursor = conn.cursor()
                cursor.execute(sql)
                result = cursor.fetchone()
                if result is not None:
                    col_name_val_tuples.append((column_name, result[0]))
                else:
                    col_name_val_tuples.append((column_name, "NULL"))
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                col_name_val_tuples.append((column_name, "NULL"))
        result_dict[table_name] = col_name_val_tuples
    
    conn.close()
    
    return result_dict

def get_example_row_for_db(db_path: str, table_columns_dict: List[str]) -> Dict[str, List[Tuple[Any]]]:
    # return a dict of table name to some rows represented as tuples
    result_dict = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    for table_name, column_names in table_columns_dict.items():
        sql = f"SELECT {', '.join(column_names)} FROM {table_name} LIMIT 3"
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            result_dict[table_name] = result
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            result_dict[table_name] = None

    return result_dict

def build_db_info(examples: Dict[str, Any], save_file_path: str):
    db_info = {}
    for example in tqdm(examples):
        db_id = example["metadata"]["db_id"]

        if db_id not in db_info:
            db_path = os.path.join(DB_DIR, f"{db_id}.db")
            db_headers = example["metadata"]["db_table_headers"]
            db_info_dict = {"db_path": db_path, "columns": db_headers, 
                            "column_example_values": get_example_val_for_db_columns(db_path, db_headers),
                            "column_example_rows": get_example_row_for_db(db_path, db_headers)}
            db_info[db_id] = db_info_dict
    
    # some basic analysis
    db_null_col_num_list = []
    for _, db_info_dict in db_info.items():
        db_null_col_num = 0
        for _, column_name_val_tuples in db_info_dict["column_example_values"].items():
            for _, column_val in column_name_val_tuples:
                if column_val == "NULL":
                    db_null_col_num += 1
        db_null_col_num_list.append(db_null_col_num)
    print(f"average number of null columns: {sum(db_null_col_num_list)/len(db_null_col_num_list)}")
                    
    
    with open(save_file_path, "w+") as f:
        json.dump(db_info, f)

def create_prompt():
    # set some constant values as function parameters
    promptify_func: callable = example_to_demonstration_sql_bridge
    save_file_path: str = "prompt_files/squall_codex_cot_sql_prompt_bridge.txt"

    # load all the examples
    results = []
    file_names = [f"data/squall/squall-codex_davinci-pass_at_50-{x}-output_gen_prob.jsonl" for x in ["train"]]
    for file_name in file_names:
        with open(file_name, "r") as f:
            results.extend([json.loads(line) for line in f.readlines()])
    question_example_dict = {example["metadata"]["db_id"]+example["metadata"]["question"]: example["metadata"] for example in results}

    # find the same few_shot examples
    with open("prompt_files/squall_codex_cot_sql_prompt_baseline.txt", "r") as f:
        lines = f.readlines()
        db_ids = [x[len("-- Database"):].strip()[:-1] for x in list(filter(lambda x: x.startswith("-- Database"), lines))]
        questions = [x[len("-- Question:"):].strip() for x in list(filter(lambda x: x.startswith("-- Question:"), lines))]
    few_shot_examples = [question_example_dict[db_id+question] for question, db_id in zip(questions, db_ids)]
    # few_shot_examples = [x["metadata"] for x in random.sample(results, 10)]

    # now actually build the prompts
    prompt = "-- Translate natural language questions into SQL queries.\n\n"
    for example in few_shot_examples:
        example_prompt = "-- Example:\n\n" + promptify_func(example)
        prompt += example_prompt + "\n\n"
    prompt = prompt.strip()
    
    with open(save_file_path, "w+") as f:
        f.write(prompt)

def get_squall_wtq_difference():
    # load all the examples from train and dev of squall, which is already prompted
    results = []
    file_names = [f"data/squall/squall-codex_davinci-pass_at_50-{x}-output_gen_prob-bridge_prompt.jsonl" for x in ["train", "dev", "wtq_only"]]
    for file_name in file_names:
        with open(file_name, "r") as f:
            results.extend([json.loads(line) for line in f.readlines()])
    id_result_dict = {x["metadata"]["metadata"]["nt"]: x["metadata"] for x in results}

    def get_table_id(file_name: str):
        # an example: csv/204-csv/622.csv
        name_splits = file_name.split("csv")
        folder_name = name_splits[1][1:-1]
        csv_name = name_splits[2][1:-1]
        assert folder_name.isnumeric()
        assert csv_name.isnumeric()
        return  f"{folder_name}_{csv_name}"

    # load the original wtq dataset, which is in tsv format
    wtq_data = []
    with open("data/squall/training.tsv", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            fields = line.strip().split("\t")
            assert len(fields) == 4

            # parse the fields
            id = fields[0]
            question = fields[1]
            table_id = get_table_id(fields[2])
            answer = fields[3]

            if id in id_result_dict:
                # if the example is in the squall dataset, then we skip it
                assert table_id == id_result_dict[id]["db_id"]
                # assert question.replace(" ", "") == id_result_dict[id]["question"].replace(" ", "") # remove all the spaces
                continue
            else:
                wtq_data.append({
                    "db_id": table_id,
                    "question": question,
                    "query": "",
                    "answer": "",
                    "original_answer": answer,
                    "metadata": {
                        "tbl": table_id, 
                        "nt": id
                    }
                })
    
    # save the wtq data
    with open("data/squall/wtq_only_train_dev_missing.jsonl", "w+") as f:
        for example in wtq_data:
            f.write(json.dumps(example) + "\n")

def get_wtq_only_db_info():
    with open("data/squall/db_info.json", "r") as f:
        db_info = json.load(f)
    
    # read all the remaining wtq examples
    with open("data/squall/wtq_only_train_dev.jsonl", "r") as f:
        wtq_only_examples = [json.loads(line) for line in f.readlines()]
    db_ids = list(filter(lambda y: y not in db_info, set([x["db_id"] for x in wtq_only_examples])))
    print(f"db_ids: {db_ids}")

    for db_id in tqdm(db_ids):
        assert db_id not in db_info

        db_path = os.path.join(DB_DIR, f"{db_id}.db")
        conn = sqlite3.connect(db_path)
        df_dict = db_to_df_dict(conn)

        db_headers = {}
        for k, v in df_dict.items():
            db_headers[k] = v.columns.tolist()

        db_info_dict = {"db_path": db_path, "columns": db_headers, 
                        "column_example_values": get_example_val_for_db_columns(db_path, db_headers),
                        "column_example_rows": get_example_row_for_db(db_path, db_headers)}
        db_info[db_id] = db_info_dict
    
    with open("data/squall/db_info_wtq.json", "w+") as f:
        json.dump(db_info, f)

def resplit_squall_to_wtq():
    # first combine the train, dev from squall and wtq only datasets
    results = []
    file_names = [f"data/squall/squall-codex_davinci-pass_at_50-{x}-output_gen_prob-bridge_prompt.jsonl" for x in ["train", "dev", "wtq_only", "wtq_only_missing"]]
    for file_name in file_names:
        with open(file_name, "r") as f:
            results.extend([json.loads(line) for line in f.readlines()])
    id_example_dict = {x["metadata"]["metadata"]["nt"]: x for x in results}
    
    # then read the original split of the wtq dataset
    with open("data/squall/random-split-1-train.tsv", "r") as f:
        wtq_train_ids = [line.split("\t")[0] for line in f.readlines()][1:]
    with open("data/squall/random-split-1-dev.tsv", "r") as f:
        wtq_dev_ids = [line.split("\t")[0] for line in f.readlines()][1:]
    
    # then split the combined dataset into train and dev
    wtq_train_prompted = [id_example_dict[i] for i in wtq_train_ids]
    wtq_dev_prompted = [id_example_dict[i] for i in wtq_dev_ids]

    print(f"wtq_train_prompted: {len(wtq_train_prompted)}")
    print(f"wtq_dev_prompted: {len(wtq_dev_prompted)}")

    with open("data/squall/wtq-codex_davinci-pass_at_50-train.jsonl", "w+") as f:
        for example in wtq_train_prompted:
            f.write(json.dumps(example) + "\n")
    with open("data/squall/wtq-codex_davinci-pass_at_50-dev.jsonl", "w+") as f:
        for example in wtq_dev_prompted:
            f.write(json.dumps(example) + "\n")

def restore_wtq_data():
    # restore the wtq data from few-shot results
    for split in ["train", "dev", "test"]:
        file_name = f"data/squall/wtq-codex_davinci-pass_at_50-{split}.jsonl"

        with open(file_name, "r") as f:
            results = [json.loads(line) for line in f.readlines()]

            with open(f"data/squall/wtq_restored_{split}.jsonl", "w+") as f:
                for result in results:
                    example = result["metadata"]
                
                    example.pop("pad_token_id")
                    example.pop("correct_token_idx")
                    example.pop("incorrect_token_idx")

                    f.write(json.dumps(example) + "\n")

def main():
    # read the data
    # data = load_json(DATA_PATH)

    # build_column_name_dict(data)

    # preprocess_squall_dataset()

    # train_dev_split()

    # all_data_aggregation()

    # results = []
    # file_names = [f"data/squall/squall-codex_davinci-pass_at_50-{x}-output_gen_prob.jsonl" for x in ["train", "dev", "test"]]
    # for file_name in file_names:
    #     with open(file_name, "r") as f:
    #         results.extend([json.loads(line) for line in f.readlines()])
    
    # build_db_info(results, "data/squall/db_info.json")

    # create_prompt()

    # get_squall_wtq_difference()

    # resplit_squall_to_wtq()

    restore_wtq_data()

if __name__ == "__main__":
    main()
    # revert_db_column_names()