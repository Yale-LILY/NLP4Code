import unittest

from os import path, sys
from typing import List, Tuple, Dict

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)


# from tests.consts import DATA_MODULES, DATASETS, FEW_SHOT_DATASETS
from tests.consts import DATASETS, FEW_SHOT_DATASETS

from torch.utils.data import DataLoader


# test cases to add:
# - test base_reader classes are abstract
# - test different modes (train, test, few_shot_test)


class TestDatasets(unittest.TestCase):
    # TODO: NotImplemented error testing
    # def test_few_shot_datasets(self):
    #     for few_shot_dataset_cls, few_shot_dataset_init_kwargs in FEW_SHOT_DATASETS:
    #         few_shot_dataset = few_shot_dataset_cls(**few_shot_dataset_init_kwargs)

    def test_finetune_datasets(self):
        for finetune_dataset_cls, finetune_dataset_init_kwargs in DATASETS:
            finetune_dataset = finetune_dataset_cls(**finetune_dataset_init_kwargs)


class TestDataModules(unittest.TestCase):
    def test_gsmath(self):
        # TODO: this is dummy test
        self.assertTrue(True)

    # def test_finetune_data_modules(self):
    #     for finetune_data_module_cls, finetune_data_module_init_kwargs in DATA_MODULES:
    #         finetune_data_module = finetune_data_module_cls(
    #             **finetune_data_module_init_kwargs
    #         )
    #         train_dl = finetune_data_module.train_dataloader()
    #         self.assertTrue(isinstance(train_dl, DataLoader))
    #         val_dl = finetune_data_module.val_dataloader()
    #         self.assertTrue(isinstance(val_dl, DataLoader))
