import unittest

from os import path, sys
from typing import List, Tuple, Dict

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)


# from tests.consts import DATA_MODULES, DATASETS, FEW_SHOT_DATASETS
from tests.consts import DATASETS, FEW_SHOT_DATASETS

from torch.utils.data import DataLoader

from finetuning.lightning_modules.datasets.base_datamodule import (
    NL2CodeDataModule,
    FewShotNL2CodeDataModule,
)


# test cases to add:
# - test base_reader classes are abstract
# - test different modes (train, test, few_shot_test)


class TestDatasets(unittest.TestCase):
    # TODO: NotImplemented error testing
    def test_few_shot_datasets(self):
        for few_shot_dataset_cls, few_shot_dataset_init_kwargs in FEW_SHOT_DATASETS:
            few_shot_dataset = few_shot_dataset_cls(
                **few_shot_dataset_init_kwargs,
            )

    def test_finetune_datasets(self):
        for finetune_dataset_cls, finetune_dataset_init_kwargs in DATASETS:
            finetune_dataset = finetune_dataset_cls(**finetune_dataset_init_kwargs)


def create_data_module_init_kwargs(
    dataset_init_kwargs: Dict, is_few_shot: bool
) -> Dict:
    data_module_init_kwargs = dataset_init_kwargs.copy()

    data_module_init_kwargs["batch_size"] = 1

    data_module_init_kwargs["val_file_path"] = data_module_init_kwargs["file_path"]
    data_module_init_kwargs["val_batch_size"] = 1
    if not is_few_shot:
        data_module_init_kwargs["train_file_path"] = data_module_init_kwargs[
            "file_path"
        ]

    del data_module_init_kwargs["file_path"]
    del data_module_init_kwargs["mode"]
    return data_module_init_kwargs


class TestDataModules(unittest.TestCase):
    def test_few_shot_data_modules(self):
        # instantiate each few shot dataset as part of a data module
        for few_shot_dataset_cls, few_shot_dataset_init_kwargs in FEW_SHOT_DATASETS:
            few_shot_dataset_cls_str = few_shot_dataset_cls.__name__
            few_shot_data_module_init_kwargs = create_data_module_init_kwargs(
                few_shot_dataset_init_kwargs, True
            )

            few_shot_data_module = FewShotNL2CodeDataModule(
                dataset_cls=few_shot_dataset_cls_str,
                **few_shot_data_module_init_kwargs,
            )

            # no train_dataloader on few shot data module
            with self.assertRaises(NotImplementedError):
                train_dl = few_shot_data_module.train_dataloader()
            val_dl = few_shot_data_module.val_dataloader()
            self.assertTrue(isinstance(val_dl, DataLoader))

    def test_finetune_data_modules(self):
        # instantiate each few shot dataset as part of a data module
        for finetune_dataset_cls, finetune_dataset_init_kwargs in DATASETS:
            finetune_dataset_cls_str = finetune_dataset_cls.__name__
            finetune_data_module_init_kwargs = create_data_module_init_kwargs(
                finetune_dataset_init_kwargs, False
            )

            finetune_data_module = NL2CodeDataModule(
                dataset_cls=finetune_dataset_cls_str,
                **finetune_data_module_init_kwargs,
            )

            train_dl = finetune_data_module.train_dataloader()
            self.assertTrue(isinstance(train_dl, DataLoader))
            val_dl = finetune_data_module.val_dataloader()
            self.assertTrue(isinstance(val_dl, DataLoader))
