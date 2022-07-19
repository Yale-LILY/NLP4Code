from overrides import overrides

from typing import Dict, Iterable, List, Any, Optional, Union

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, NL2CodeDataModule
from few_shot.codex_spider import example_to_demonstration_sql

class SpiderDataset(NL2CodeDataset):

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = example_to_demonstration_sql(example, train=False) # we only need the context
        code = example["query"]
        return self.get_example_dict(example, context, code, train_mode=True)

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = example_to_demonstration_sql(example, train=False) # we only need the context
        code = example["query"]
        return self.get_example_dict(example, context, code, train_mode=False)

class Text2SqlDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate", "test"]

        train_data = SpiderDataset(file_path=self.train_file_path,
                                   transformer_model_name=self.transformer_model_name,
                                   max_instances=self.train_max_instances, 
                                   mode="train", few_shot_n=self.few_shot_n)
        self.train_data = train_data

        val_data = SpiderDataset(file_path=self.val_file_path,
                                 transformer_model_name=self.transformer_model_name,
                                 max_instances=self.val_max_instances, 
                                 mode="test", few_shot_n=self.few_shot_n)
        self.val_data = val_data 

def get_gold_program_func(example_dict: Dict[str, Any]):
    return example_dict["metadata"]["query"]

def get_gold_answer_func(example_dict: Dict[str, Any]):
    return example_dict["metadata"]["answer"]