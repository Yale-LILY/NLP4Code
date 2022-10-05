import openai
import os
import json
import random
import time
import sqlite3

random.seed(333)

from typing import List, Tuple, Any, Dict, Text
from tqdm import tqdm

from execution.spider_execution import connect_databse, db_to_df_dict, spider_execution_py
from execution.spider_execution import spider_execution_sql, spider_answer_eq

manually_annotated_sql_to_py = {
    # train-7
    "SELECT DISTINCT T1.creation FROM department AS T1 JOIN management AS T2 ON T1.department_id  =  T2.department_id JOIN head AS T3 ON T2.head_id  =  T3.head_id WHERE T3.born_state  =  'Alabama'": \
        "t1 = pd.merge(department, management, on='department_id')\nt2 = pd.merge(t1, head, on='head_id')\nt3 = t2[t2['born_state'] == 'Alabama']\nanswer = t3['creation'].unique()",
    # train-15
    "SELECT T1.department_id ,  T1.name ,  count(*) FROM management AS T2 JOIN department AS T1 ON T1.department_id  =  T2.department_id GROUP BY T1.department_id HAVING count(*)  >  1": \
        "t1 = pd.merge(management, department, left_on='department_id', right_on='department_id')\nt2 = t1.groupby(['department_id', 'name']).size().rename('count')\nanswer = t2[t2 > 1].to_frame().reset_index()",
    # train-9
    "SELECT creation FROM department GROUP BY creation ORDER BY count(*) DESC LIMIT 1": \
        "t1 = department.groupby('creation').size().rename('count')\nt2= t1.sort_values(ascending=False).head(1).to_frame().reset_index()\nanswer = t2['creation']",
    # train-35
    "SELECT T2.Year ,  T1.Official_Name FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID": \
        "t1 = pd.merge(city, farm_competition, left_on='city_id', right_on='host_city_id')\nanswer = t1[['year', 'official_name']]",
}

def saved_promptify_sql(prompt_file_path: str, example: Dict, max_prompt_examples: int = 100, lower_case_schema: bool = False) -> Text:
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()

    # cut by the max prompt examples
    prompt_examples = prompt.split('\n\n-- Example:')
    assert len(prompt_examples) == 9
    prompt = '\n\n-- Example'.join(prompt_examples[:max_prompt_examples+1]).strip()

    example_text = example_to_demonstration_sql(example, train=False, lower_case_schema=lower_case_schema)
    prompt += '\n\n-- Example:\n\n' + example_text.strip()

    return prompt

def example_to_demonstration_sql_3(example: Dict, train: bool = True) -> Text:
    text = f"{example['question']} | {example['db_id']} |"
    
    for table_name, columns in example['db_table_headers'].items():
        column_representation = ' , '.join(columns)
        text += f' {table_name} : {column_representation} |'

    if train:
        text += f' {example["query"]}'
    else:
        text += ' '

    return text

def example_to_demonstration_sql_2(example: Dict, train: bool = True) -> Text:
    text = f"| {example['db_id']} |"
    for table_name, columns in example['db_table_headers'].items():
        column_representation = ' , '.join(columns)
        text += f' {table_name} : {column_representation} |'
    
    text += f' {example["question"]} |'

    if train:
        text += f' {example["query"]}'
    else:
        text += ' '

    return text

def example_to_demonstration_sql(example: Dict, train: bool = True, lower_case_schema: bool = False) -> Text:
    text = f'-- Database {example["db_id"]}:\n'
    for table_name, columns in example['db_table_headers'].items():
        column_representation = ', '.join(columns)
        text += f'--  Table {table_name}: {column_representation}\n'
    
    if lower_case_schema:
        text = text.lower() + f'-- question: {example["question"]}\n'
    else:
        text += f'-- Question: {example["question"]}\n'

    if train:
        text += f'-- SQL:\n{example["query"]}'
    else:
        text += '-- SQL:\n'

    return text


def promptify_sql(prompt_examples: List[Dict], example: Dict) -> Text:
    prompt = '-- Translate natural language questions into SQL queries.\n'

    for idx, prompt_example in enumerate(prompt_examples, 1):
        example_text = example_to_demonstration_sql(prompt_example)
        example_text = f"""-- Example:

{example_text}"""

        prompt += '\n' + example_text + '\n'

    example_text = example_to_demonstration_sql(example, train=False)
    prompt += f"""
-- Example:

{example_text}"""

    return prompt


def example_to_demonstration_python(example: Dict, train: bool = True, code_as_is: bool = False) -> Text:
    text = f'# Dataset {example["db_id"]}:\n\n'
    
    for table_name, columns in example['db_table_headers'].items():
        column_representation = ', '.join(columns)
        text += f"""# DataFrame {table_name}: {column_representation.lower()}
{table_name} = df_dict['{table_name}']\n"""
    
    text += '\n' + f'# Question: {example["question"]}\n'

    if train:
        if code_as_is:
            text += f'{example["pandas_converted"]}\n\n'
        else:
            text += f'answer = {example["pandas_converted"]}\n\n'
    else:
        text += ''

    return text


def promptify_python(prompt_examples: List[Dict], example: Dict) -> Text:
    prompt = '# Translate natural language questions into Pandas programs.\n'

    for idx, prompt_example in enumerate(prompt_examples, 1):
        code_as_is = "answer = " in prompt_example["pandas_converted"]
        example_text = example_to_demonstration_python(prompt_example, code_as_is=code_as_is)
        prompt += '\n' + example_text + '\n'

    example_text = example_to_demonstration_python(example, train=False)
    prompt += '\n' + example_text

    return prompt

def generate(input: str, engine: str, max_tokens: int, **kwargs) -> List[str]:

    completion = openai.Completion.create(engine=engine, prompt=input, max_tokens=max_tokens, **kwargs)
    
    return completion.choices[0].text


def openai_call(input: List[str], engine: str="code-davinci-001", max_tokens: int=1024, **kwargs) -> List[List[str]]:
    
    api_key = os.environ.get('OPENAI_API_KEY')

    if api_key is None:
        raise Exception("Please set your OpenAI API key in the environment variable OPENAI_API_KEY")
    else:
        openai.api_key = api_key

    completions = []

    for prompt in input:
        output = generate(prompt, engine, max_tokens=max_tokens, **kwargs)
        completions.append(output)
    
    return completions

def codex(input: List[str], engine: str="code-davinci-001", max_tokens: int=1024, **kwargs) -> List[List[str]]:
    return openai_call(input, engine, max_tokens, **kwargs)

def gpt3(input: List[str], engine: str="text-davinci-001", max_tokens: int=1024, **kwargs) -> List[List[str]]:
    return openai_call(input, engine, max_tokens, **kwargs)

def select_few_shot_examples(examples: List[Dict[str, Any]], n: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:   
    few_shot_example_queries = []
    few_shot_example_idx = []

    # first round, only pick the annotated ones
    for i, example in enumerate(examples):
        if example["query"] in manually_annotated_sql_to_py and example["query"] not in few_shot_example_queries:
            example["pandas_converted"] = manually_annotated_sql_to_py[example["query"]]
            few_shot_example_idx.append(i)
            few_shot_example_queries.append(example["query"])

    for i, example in enumerate(examples):
        # must be less than 5 tables
        if len(example['db_table_headers']) < 5 and len(few_shot_example_idx) < n \
                and "pandas_converted" in example and example["query"] not in few_shot_example_queries:
            few_shot_example_idx.append(i)
            few_shot_example_queries.append(example["query"])

    few_shot_examples = []
    test_examples = []
    for i in range(len(examples)):
        if i in few_shot_example_idx:
            few_shot_examples.append(examples[i])
        else:
            test_examples.append(examples[i])

    return few_shot_examples, test_examples 

def test_spider_few_shot(train_examples: List[Dict[str, Any]], dev_examples: List[Dict[str, Any]]) -> bool:
    # split the data into few-shot examples and test examples
    few_shot_examples, _ = select_few_shot_examples(train_examples, n=10)
    test_examples = dev_examples
    random.shuffle(few_shot_examples)
    random.shuffle(test_examples)

    # save the results to file line by line
    f = open("spider_codex_sql_py_10_shot_results.jsonl", "w+")

    sql_result_list = []
    py_result_list = []
    for example in tqdm(test_examples[:50]):
        # get two versions of the database
        db_file_path = f"data/spider/database/{example['db_id']}/{example['db_id']}.sqlite"
        conn = connect_databse(db_file_path)
        try:
            df_dict = db_to_df_dict(conn)
        except sqlite3.OperationalError as e:
            print("skip example with unsucessful db_to_df_dict conversion")
            continue

        # get the two versions of the prompt
        sql_prompt = promptify_sql(few_shot_examples, example)
        py_prompt = promptify_python(few_shot_examples, example)

        # send to codex for completion
        while True:
            try:
                sql_results = codex([sql_prompt], engine="code-davinci-001", max_tokens=200, temperature=0.2)
                py_results = codex([py_prompt], engine="code-davinci-001", max_tokens=200, temperature=0.2)
                break
            except Exception as e:
                print(f"Codex failed to complete, retrying - {str(e)}")
                time.sleep(5)
                continue
    
        generated_sql = sql_results[0].split('\n')[0]
        generated_py = py_results[0].split('\n\n')[0]

        assert len(sql_results) == 1
        assert len(py_results) == 1

        sql_exec_result = spider_execution_sql(generated_sql, conn, return_error_msg=True)
        py_exec_result = spider_execution_py(generated_py, df_dict, return_error_msg=True)

        if isinstance(sql_exec_result, str) and sql_exec_result.startswith("ERROR"):
            sql_error = sql_exec_result
            sql_exec_result = None
        else:
            sql_error = None
        if isinstance(py_exec_result, str) and py_exec_result.startswith("ERROR"):
            py_error = py_exec_result
            py_exec_result = None
        else:
            sql_error = None

        sql_result_list.append(spider_answer_eq(sql_exec_result, example["answer"]))
        py_result_list.append(spider_answer_eq(py_exec_result, example["answer"]))

        result_dict = {
            "generated_sql": generated_sql,
            "generated_py": generated_py,
            "sql_error": sql_error,
            "py_error": py_error,
            "sql_exec_result": str(sql_exec_result),
            "py_exec_result": str(py_exec_result),
            "gold_answer": example["answer"],
            "gold_sql": example["query"],
            "table_headers": example["db_table_headers"],
            "gold_py": example["pandas_converted"] if "pandas_converted" in example else None,
        }

        example["results"] = result_dict
        f.write(json.dumps(example) + '\n')

    f.close()
    print(f"sql exec accuracy: {sum(sql_result_list) / len(sql_result_list)}")
    print(f"py exec accuracy: {sum(py_result_list) / len(py_result_list)}")


def main():
    # load a jsonl file
    with open("data/spider/train_spider_processed_v2.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]

    with open("data/spider/dev_processed.jsonl", "r") as f:
        dev_data = [json.loads(line) for line in f]

    # test the few shot
    test_spider_few_shot(train_data, dev_data)

if __name__ == "__main__":
    # main()

    # exit(0)

    with open("spider_codex_sql_py_10_shot_results.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

        sql_acc_accum, sql_rate = 0, 0
        py_acc_accum, py_rate = 0, 0
        sql_over_py_idx_list = []
        py_over_sql_idx_list = []
        for i, example in enumerate(data):
            # get two versions of the database
            db_file_path = f"data/spider/database/{example['db_id']}/{example['db_id']}.sqlite"
            conn = connect_databse(db_file_path)
            df_dict = db_to_df_dict(conn)

            sql_exec_result = spider_execution_sql(example["results"]["generated_sql"], conn)
            py_exec_result = spider_execution_py(example["results"]["generated_py"], df_dict)

            sql_rate += int(sql_exec_result is not None)
            py_rate += int(py_exec_result is not None)

            sql_acc = int(spider_answer_eq(sql_exec_result, example["answer"]))
            py_acc = int(spider_answer_eq(py_exec_result, example["answer"]))

            if sql_acc > py_acc:
                sql_over_py_idx_list.append(i)
            elif py_acc > sql_acc:
                py_over_sql_idx_list.append(i)

            example["results"]["sql_accuracy"] = int(sql_acc)
            example["results"]["py_accuracy"] = int(py_acc)

            sql_acc_accum += float(sql_acc)
            py_acc_accum += float(py_acc)
        
        print(f"sql exec accuracy: {sql_acc_accum / len(data)}")
        print(f"py exec accuracy: {py_acc_accum / len(data)}")

        print(f"sql rate: {sql_rate / len(data)}")
        print(f"py rate: {py_rate / len(data)}")

        print(f"sql over py: {sql_over_py_idx_list}")
        print(f"py over sql: {py_over_sql_idx_list}")

    # save the results to file line by line
    # with open("spider_codex_sql_py_few_shot_results_eval.jsonl", "w+") as f:
    #     for example in data:
    #         f.write(json.dumps(example) + '\n')
        

