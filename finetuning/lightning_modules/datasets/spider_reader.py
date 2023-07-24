import re
import os
import pandas as pd
import json

from overrides import overrides

from typing import Dict, Iterable, List, Any, Optional, Union, Tuple

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, FewShotNL2CodeDataset

# DB_INFO_FILE = os.path.join(os.path.dirname(__file__), '../../../data/squall/db_info_wtq.json')
DB_INFO_FILE = os.path.join(os.path.dirname(__file__), f"{os.environ['NLP4CODE_TEST_DATA_PATH']}/squall/db_info_wtq.json")
with open(DB_INFO_FILE, "r") as f:
    full_db_info = json.load(f)


class FewShotSpiderDataset(FewShotNL2CodeDataset):

    instruction: str = "-- Given database schema and a question in natural language, generate the corresponding SQL query."
    full_db_info = None
    DB_INFO_FILE = os.path.join(os.path.dirname(__file__), '../../../data/squall/db_info_wtq.json')
    example_io_sep: str = "\n"

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = self.get_prompt_for_example(example)

        return [self.get_example_dict(example, context, train_mode=False)]
    
    # @overrides
    def promptify_example(self, example: Dict[str, Any], add_code: bool = True, use_bridge_format: bool = False) -> Tuple[str, str]:
        # check if the DB info file is loaded
        if self.full_db_info is None and use_bridge_format:
            with open(self.DB_INFO_FILE, "r") as f:
                self.full_db_info = json.load(f)
        
        # add db schema
        db_id = example['db_id']
        text = f'-- Database {db_id}:\n'
        if use_bridge_format:
            db_info = self.full_db_info[db_id]
            for table_name, columns in db_info['column_example_values'].items():
                column_representation = ', '.join([f"{name} ({str(val)[:50] + ('...' if len(str(val)) > 50 else '')})" for name, val in columns])
                text += f'--  Table {table_name}: {column_representation}\n'
        else:
            for table_name, columns in example['db_table_headers'].items():
                column_representation = ', '.join(columns)
                text += f'--  Table {table_name}: {column_representation}\n'
        
        # finalize the context and the code
        context = text + f'-- Question: {example["question"]}\n-- SQL:'
        code = example["query"] if add_code else ""

        return context, code

class SpiderDataset(NL2CodeDataset):

    def __init__(self, promptify_func: str = "example_to_demonstration_sql", use_distinct: bool = False, 
                 use_decomp_sql: bool = False, use_skg_format: bool = False, **kwargs):
        # init some spider specific processing options
        self.promptify_func = eval(promptify_func)
        self.use_distinct = use_distinct
        self.use_skg_format = use_skg_format

        if use_decomp_sql:
            raise NotImplementedError("DecompSQL has been deprecated")

        super().__init__(**kwargs)

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:

        if self.use_skg_format:
            context = example["text_in"] + example["struct_in"]
            code: str = example["seq_out"]
            example["db_path"] = os.path.join("data/spider/database", example["db_id"], f'{example["db_id"]}.sqlite')
            example["answer"] = -10000 # this is a dummy value
        else:
            context = self.promptify_func(example, train=False, lower_case_schema=True) # we only need the context
            code: str = example["query"]

            # lower everything but the ones in the quote
            code = code.replace("\"", "'")
            code = re.sub(r"\b(?<!')(\w+)(?!')\b", lambda match: match.group(1).lower(), code) 

        return [self.get_example_dict(example, context, code, train_mode=True)]

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:

        if self.use_skg_format:
            context = example["text_in"] + example["struct_in"]
            code: str = example["seq_out"]

            example["db_path"] = os.path.join("data/spider/database", example["db_id"], f'{example["db_id"]}.sqlite')
            example["answer"] = -10000 # this is a dummy value since we don't need the answer to eval for spider
        else:
            context = self.promptify_func(example, train=False, lower_case_schema=True) # we only need the context
        
        return [self.get_example_dict(example, context, train_mode=False)]