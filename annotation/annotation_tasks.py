import json
import os
import random
import sqlite3
import time

import pandas as pd

from overrides import overrides
from typing import List, Dict, Any, Tuple

from execution.executors import SpiderExecutor, WTQExecutor, WTQPythonExecutor
from execution.spider_execution import spider_execution_pd_sql, pd_df_to_dict, spider_execution_py, db_to_df_dict

class AnnotationTask:
    def __init__(self, data_file, output_file):
        self.data_file = data_file
        self.output_file = output_file

        # read the examples to annotate
        self.examples = []
        self.read_data()
        self.postprocess_data()

        # create the output file if it does not exist
        if not os.path.isfile(output_file):
            self.annotation_indices = self.get_annotation_indices()
            print(f"Creating new output file {output_file} for {len(self.annotation_indices)} examples")
            with open(output_file, "w+") as f:
                annotation_metadata = {"data_file": self.data_file, 
                                       "total_num_examples": len(self.examples),
                                       "annotation_indices": self.annotation_indices}
                f.write(json.dumps(annotation_metadata) + "\n")

            # keep track of the progress
            self.annotated_examples = []
        else:
            # recovery the metadata and the annotated examples
            self.recovery_progress(output_file)
    
    def postprocess_data(self):
        return

    def read_data(self):
        with open(self.data_file, "r") as f:
            self.examples = [json.loads(s) for s in f.readlines()]
    
    def get_annotation_indices(self) -> List[int]:
        return list(range(len(self.examples)))

    def recovery_progress(self, output_file: str):
        # first recover the annotation progress from the file
        with open(output_file, "r") as f:
            lines = f.readlines()
            annotation_metadata = json.loads(lines[0])
            self.annotated_examples = [json.loads(s) for s in lines[1:]]
            self.annotation_indices = annotation_metadata["annotation_indices"]
        
        # then verify the annotated examples to match the data file
        for i, example in enumerate(self.annotated_examples):
            assert example["metadata"] == self.examples[self.annotation_indices[i]], \
                f"Annotated example does not match the data file"
        
        print(f"Recovered progress from {output_file} for {len(self.annotated_examples)} out of {len(self.annotation_indices)} total examples")
        time.sleep(2)
    
    def save_single_annotation(self, example: Dict[str, Any], annotation: str, exec_result: Any = None):
        save_example = {"metadata": example, "annotation": annotation, "exec_result": exec_result}

        # save both to the output file and the annotated examples
        self.annotated_examples.append(save_example)
        with open(self.output_file, "a") as f:
            f.write(json.dumps(save_example) + "\n")
    
    def get_and_display_next_example(self):
        next_example_idx = self.annotation_indices[len(self.annotated_examples)]
        print("\033[1;7;34m" + '#' * 20 + f" Example {next_example_idx} " + '#' * 20 + "\033[0m")
        self.display_example(self.examples[next_example_idx])
        print("\033[1;7;34m" + '#' * 40 + "\033[0m")

        return self.examples[next_example_idx]
    
    def display_example(self, example: Dict[str, Any]):
        raise NotImplementedError("Please implement this method in the subclass")

    def check_annotation_correctness(self, example: Dict[str, Any], annotation: str) -> Tuple[bool, str]:
        raise NotImplementedError("Please implement this method in the subclass")
    
    def get_annotation_instructions(self, example: Dict[str, Any]) -> str:
        return "Enter annotation (or `exit`/`skip`): "

class SQL2PandasAnnotationTask(AnnotationTask):
    def __init__(self, dataset_name: str, annotation_size: int=100):
        # init the parameters
        assert dataset_name in ["spider", "squall"], f"Invalid dataset name {dataset_name}"
        self.dataset_name = dataset_name
        self.annotation_size = annotation_size
        self.executor = WTQPythonExecutor() if dataset_name == "squall" else None

        data_file_name = "data/spider/train_spider_processed_v2.jsonl" if dataset_name == "spider" \
            else "data/squall/squall_processed_train_all.jsonl"
        output_file_name = f"{os.getlogin()}_{dataset_name}_annotation.jsonl"

        # init the base class
        super().__init__(data_file_name, output_file_name)

    @overrides
    def postprocess_data(self):
        # add db_path to the examples
        for example in self.examples:
            if self.dataset_name == "spider":
                example["db_path"] = os.path.join("data/spider/database", example["db_id"], f'{example["db_id"]}.sqlite')
            elif self.dataset_name == "squall":
                example["db_path"] = os.path.join("data/squall/tables/db", example["db_id"] + ".db")
            else:
                raise ValueError(f"Unknown dataset name {self.dataset_name}")
        
    @overrides
    def get_annotation_indices(self) -> List[int]:
        all_indices = list(range(len(self.examples)))
        random.shuffle(all_indices)

        return all_indices[:self.annotation_size]
    
    def display_database(self, db_path: str):
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df_dict = {}
        for table_name in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            df_dict[table_name[0]] = pd.read_sql_query(f"SELECT * FROM {table_name[0]}", conn)
        
        for table_name, df in df_dict.items():
            print('-' * 50)
            print(f"Table: {table_name}, Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            df_to_print = df.head(5)
            print(df_to_print)
            print('-' * 50)
            print("NOTE: Only the first 5 rows are shown!!!")
    
    @overrides
    def display_example(self, example: Dict[str, Any]) -> str:
        print("Database:")
        self.display_database(example['db_path'])
        print(f"Question: {example['question']}")
        print(f"SQL Query: {example['query']}")

    @overrides
    def check_annotation_correctness(self, example: Dict[str, Any], annotation: str) -> Tuple[bool, str]:
        annotated_sql = annotation.strip()
        exec_match, exec_result = self.executor.exec_program(annotated_sql, example)
        if exec_match == 1:
            return True, f"{exec_result}"
        else:
            expected_answer = example['answer'] if self.dataset_name == "spider" else example['original_answer']
            return False, f"Expected: {expected_answer} but got {exec_result}"

    @overrides
    def get_annotation_instructions(self, example: Dict[str, Any]) -> str:
        basic_prompt = "Enter annotation (or `exit`/`skip`), for python use `;` to seperate lines:\n"

        # construct the example specific prompt
        conn = sqlite3.connect(example["db_path"])
        df_dict = db_to_df_dict(conn)
        table_vars_code = "import pandas as pd\n"
        for table_name in df_dict.keys():
            table_vars_code += f"# {' '.join(list(df_dict[table_name].columns))}\n{table_name} = df_dict['{table_name}']\n"
        example_prompt = "; ".join(list(filter(lambda x: not x.startswith("#"), table_vars_code.split("\n"))))

        return basic_prompt + example_prompt

