import json
import random
import sqlite3
import pandas as pd

random.seed(333)

from typing import Dict, List, Optional, Any, Callable, Tuple
from codex import codex_evaluate_pass_at_k
from codex_spider import select_few_shot_examples
from execution.spider_execution import spider_execution_py, spider_answer_eq, connect_databse, db_to_df_dict

def promptify_sql2py(example: Dict[str, Any], include_sol: bool) -> str:
    text = f'# Dataset {example["db_id"]}:\n\n'
    
    for table_name, columns in example['db_table_headers'].items():
        column_representation = ', '.join(columns)
        text += f"""# DataFrame {table_name}: {column_representation.lower()}\n{table_name} = df_dict['{table_name}']\n"""
    
    text += '\n' + f'# Question: {example["question"]}\n'
    text += f'# SQL: {example["query"].lower()}\n'

    if include_sol:
        text += f'{example["pandas_converted"]}\n\n'

    return text

def get_df_dict_from_example(example: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    try:
        db_file_path = f"data/spider/database/{example['db_id']}/{example['db_id']}.sqlite"
        conn = connect_databse(db_file_path)
        df_dict = db_to_df_dict(conn)
    except sqlite3.OperationalError as e:
        print("skip example with unsucessful db_to_df_dict conversion")
        df_dict = None

    return df_dict

def spider_prepare_extra_exec_args_py(examples: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {"df_dict": [get_df_dict_from_example(example) for example in examples]}

def verify_py(example: Dict[str, Any], py_code: str) -> bool:
    df_dict = get_df_dict_from_example(example)
    answer = example["answer"]

    py_exec_result = spider_execution_py(py_code, df_dict, return_error_msg=True)
    return py_exec_result

def iterative_codex_few_shot_conversion():
    print("Loading datasets...\n")
    with open("data/spider/train_spider_processed_v2.jsonl", "r") as f:
        training_data = [json.loads(line) for line in f.readlines()]
        random.shuffle(training_data)

    n_examples = 100
    few_shot_n = 2
    # ks = [5, 10, 20, 50, 100]
    ks = [20]
    temp = 0.8

    # use some examples in the training data with converted pandas code as the few shot examples
    few_shot_instances, test_instances = select_few_shot_examples(training_data, n=few_shot_n)
    test_instances = test_instances[:n_examples]

    # make sure all the test instances can have the df_dict
    test_instances = list(filter(lambda x: get_df_dict_from_example(x) is not None, test_instances))[:n_examples]

    # preprocess the pandas converted code
    for example in few_shot_instances:
        example["pandas_converted"] = example["pandas_converted"].strip()
        if "answer = " not in example["pandas_converted"]:
            assert "\n" not in example["pandas_converted"]
            example["pandas_converted"] = "answer = " + example["pandas_converted"]

    print(f"Evaluating {few_shot_n}-shot pass@{str(ks)} performance on {n_examples} examples...")

    codex_evaluate_pass_at_k(few_shot_examples=few_shot_instances,
                            input_dataset=test_instances,
                            text_header="# Translating the following SQL queries to Pandas code:",
                            promptify_func=promptify_sql2py,
                            exec_func=spider_execution_py,
                            answer_eq_func=spider_answer_eq,
                            prepare_extra_exec_args_func=spider_prepare_extra_exec_args_py,
                            eval_at_ks=ks,
                            openai_kwargs={"temperature": temp, "engine": "code-davinci-002"},
                            save_result_path=f"spider_codex_davinci2_{few_shot_n}_shot_pass_at_{str(ks)}results_{n_examples}_temp_{temp}.jsonl",
                            batch_prompts=5)

def conversion_analysis():
    with open("spider_codex_davinci2_2_shot_pass_at_[20]results_100_temp_0.8.jsonl", "r") as f:
        results = json.load(f)
    
    for result in results:
        if result["pass@20"] < 1.0:
            print(result["example"]["query"])
            if len(result["example"]["query"].split(" ")) < 15:
                print()

if __name__ == "__main__":
    codex_few_shot_conversion()
    # conversion_analysis()