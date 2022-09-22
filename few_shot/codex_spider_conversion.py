import json
import random
import sqlite3
import pandas as pd

random.seed(333)

from typing import Dict, List, Optional, Any, Callable, Tuple
from codex import codex_evaluate_pass_at_k
from codex_spider import get_few_shot_annotated_examples, select_few_shot_examples, manually_annotated_sql_to_py
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

def iterative_codex_few_shot_conversion(examples: List[Dict[str, Any]], 
                                        ks_list: List[int] = [50, 100],
                                        max_samples_per_request: int = 100,
                                        temps_list = [0.8, 1.0]):
    # use some examples in the training data with converted pandas code as the few shot examples
    few_shot_instances = get_few_shot_annotated_examples(examples)
    print(f"{len(few_shot_instances)} few shot instances are used")
    assert len(few_shot_instances) > 0

    # make sure all the test instances can have the df_dict
    conversion_instances = examples
    # conversion_instances = list(filter(lambda x: get_df_dict_from_example(x) is not None, examples))
    # print(f"{len(examples) - len(conversion_instances)}/{len(examples)} examples" + \
    #         "can not be converted since extract df_dict failed")

    for k, temp in zip(ks_list, temps_list):
        print(f"Try to convert {len(conversion_instances)} examples using k={k}")

        generated_programs, exec_match_list = \
            codex_evaluate_pass_at_k(few_shot_examples=few_shot_instances,
                                     input_dataset=conversion_instances,
                                     text_header="# Translating the following SQL queries to Pandas code:",
                                     promptify_func=promptify_sql2py,
                                     exec_func=spider_execution_py,
                                     answer_eq_func=spider_answer_eq,
                                     prepare_extra_exec_args_func=spider_prepare_extra_exec_args_py,
                                     eval_at_ks=[k],
                                     openai_kwargs={"temperature": temp, "engine": "code-davinci-002"},
                                     save_result_path=None,
                                     batch_prompts=(max(min(int(max_samples_per_request / k), 20), 1)))

        assert len(generated_programs) == len(exec_match_list)
        assert all([len(generated_programs[i]) == len(exec_match_list[i]) for i in range(len(generated_programs))])

        # save the results
        next_iter_instances = []
        with open(f"few_shot_results/spider_codex_conversion_k_{k}_n_{len(conversion_instances)}.jsonl", "w+") as f:
            for i in range(len(generated_programs)):
                program_result_list = list(zip(generated_programs[i], exec_match_list[i]))
                save_dict = {"example": conversion_instances[i], "program_result_list": program_result_list}
                f.write(json.dumps(save_dict) + "\n")

                if sum(exec_match_list[i]) == 0:
                    next_iter_instances.append(conversion_instances[i])
        
        # go into the next iteration
        print(f"{len(next_iter_instances)}/{len(conversion_instances)} left unconverted")
        conversion_instances = next_iter_instances

def conversion_analysis():
    with open("spider_codex_davinci2_2_shot_pass_at_[20]results_100_temp_0.8.jsonl", "r") as f:
        results = json.load(f)
    
    for result in results:
        if result["pass@20"] < 1.0:
            print(result["example"]["query"])
            if len(result["example"]["query"].split(" ")) < 15:
                print()

if __name__ == "__main__":
    print("Loading datasets...\n")
    with open("data/spider/train_spider_processed_v2.jsonl", "r") as f:
        training_data = [json.loads(line) for line in f.readlines()]
        few_shot_examples = list(filter(lambda x: x['query'] in manually_annotated_sql_to_py, training_data))
        print(f"{len(few_shot_examples)} few-shot examples are recovered")
    
    with open("few_shot_results/spider_codex_conversion_k_20_n_1969.jsonl", "r") as f:
        results = [json.loads(line) for line in f.readlines()]
        failed_examples = list(filter(lambda x: sum(map(lambda y: y[1], x["program_result_list"])) == 0, results))
        training_data = list(map(lambda x: x["example"], failed_examples)) + few_shot_examples

    iterative_codex_few_shot_conversion(training_data)