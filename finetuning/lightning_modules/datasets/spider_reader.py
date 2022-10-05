import re
import os
import pandas as pd

from overrides import overrides

from typing import Dict, Iterable, List, Any, Optional, Union

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, NL2CodeDataModule
from few_shot.codex_spider import example_to_demonstration_sql, saved_promptify_sql
from preprocessing.preprocess_spider import decompose_sql, pd_df_from_dict

class FewShotSpiderDataset(NL2CodeDataset):

    def __init__(self, 
                 prompt_file: str,
                 prompt_examples: int = 100, 
                 **kwargs):
        # init some dataset specific variables
        self.prompt_examples = prompt_examples
        self.prompt_file = prompt_file

        super().__init__(**kwargs)

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise ValueError("Few shot datasets do not support training")

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = saved_promptify_sql(self.prompt_file, example, max_prompt_examples=self.prompt_examples)

        return [self.get_example_dict(example, context, train_mode=False)]

class SpiderDataset(NL2CodeDataset):

    def __init__(self, use_distinct: bool = False, use_decomp_sql: bool = False, use_skg_format: bool = False, **kwargs):
        # init some spider specific processing options
        self.use_distinct = use_distinct
        self.use_skg_format = use_skg_format

        if use_decomp_sql:
            raise NotImplementedError("DecompSQL has been deprecated")

        super().__init__(**kwargs)

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        example["db_path"] = os.path.join("data/spider/database", example["db_id"], f'{example["db_id"]}.sqlite')

        if self.use_skg_format:
            context = example["text_in"] + example["struct_in"]
            code: str = example["seq_out"]
            example["answer"] = -10000 # this is a dummy value
        else:
            context = example_to_demonstration_sql(example, train=False, lower_case_schema=True) # we only need the context
            code: str = example["query"]

            # lower everything but the ones in the quote
            code = code.replace("\"", "'")
            code = re.sub(r"\b(?<!')(\w+)(?!')\b", lambda match: match.group(1).lower(), code) 

        return [self.get_example_dict(example, context, code, train_mode=True)]

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        example["db_path"] = os.path.join("data/spider/database", example["db_id"], f'{example["db_id"]}.sqlite')

        if self.use_skg_format:
            context = example["text_in"] + example["struct_in"]
            code: str = example["seq_out"]

            example["answer"] = -10000 # this is a dummy value since we don't need the answer to eval for spider
        else:
            context = example_to_demonstration_sql(example, train=False, lower_case_schema=True) # we only need the context
        
        return [self.get_example_dict(example, context, train_mode=False)]

class FewShotSQLDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            raise ValueError("Few shot datasets do not support training")

        if self.val_data is None:
            val_data = FewShotSpiderDataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 

class Text2SqlDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            train_data = SpiderDataset(transformer_model_name=self.transformer_model_name,
                                    mode="train", **self.train_set_init_args)
            self.train_data = train_data

        val_data = SpiderDataset(transformer_model_name=self.transformer_model_name,
                                 mode="test", **self.val_set_init_args)
        self.val_data = val_data 