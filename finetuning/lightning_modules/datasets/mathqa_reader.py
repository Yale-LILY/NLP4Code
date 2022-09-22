from typing import Dict, Iterable, List, Any, Optional, Union
from overrides import overrides

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, NL2CodeDataModule

class MathQADataset(NL2CodeDataset):

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.get_example_dict(example, example["text"], example["code"], train_mode=True)]

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.get_example_dict(example, example["text"], example["code"], train_mode=False)]

class MathQADataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate", "test"]

        train_data = MathQADataset(file_path=self.train_file_path,
                                   transformer_model_name=self.transformer_model_name,
                                   max_instances=self.train_max_instances, 
                                   mask_context_loss=self.mask_context_loss,
                                   mode="train", few_shot_n=self.few_shot_n)
        self.train_data = train_data

        val_data = MathQADataset(file_path=self.val_file_path,
                                 transformer_model_name=self.transformer_model_name,
                                 max_instances=self.val_max_instances, 
                                 mask_context_loss=self.mask_context_loss,
                                 mode="test", few_shot_n=self.few_shot_n)
        self.val_data = val_data 