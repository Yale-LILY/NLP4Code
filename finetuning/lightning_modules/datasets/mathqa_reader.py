from typing import Dict, Iterable, List, Any, Optional, Union
from overrides import overrides

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, NL2CodeDataModule

class MathQADataset(NL2CodeDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        train_data = MathQADataset(transformer_model_name=self.transformer_model_name,
                                   mode="train",
                                   **self.train_set_init_args)
        self.train_data = train_data

        val_data = MathQADataset(transformer_model_name=self.transformer_model_name,
                                 mode="test",
                                 **self.val_set_init_args)
        self.val_data = val_data 