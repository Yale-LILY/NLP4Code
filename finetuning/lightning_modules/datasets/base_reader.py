import json
import logging
import sys
import os
import torch

from typing import Dict, Iterable, List, Any, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from finetuning.lightning_modules.models.seq2seq_model_util import get_model, left_pad_sequences
from execution.program_tracing import get_state_repr, is_trivial_state

from torch.utils.data import DataLoader

# set environment variable to avoid deadlocks, see: 
# https://docs.allennlp.org/main/api/data/data_loaders/multiprocess_data_loader/#multiprocessdataloader.common_issues
os.environ['TOKENIZERS_PARALLELISM']='0'

logger = logging.getLogger(__name__)

FEW_SHOT_RESERVED = 10


class NL2CodeDataset(Dataset):

    def __init__(
        self, 
        file_path: str,
        transformer_model_name: str, 
        max_instances: int,
        few_shot_n: int = 0,
        mode: str = "train", 
        multi_example_instance: bool = False,
        **kwargs):
        super().__init__(**kwargs)

        # mode is one of ["train", "test", "test_few_shot"]
        assert mode in ["train", "test", "test_few_shot"]

        _, self.tokenizer = get_model(transformer_model_name, tokenizer_only=True)

        self.max_instances = max_instances
        self.mode = mode
        self.multi_example_instance = multi_example_instance

        assert few_shot_n <= FEW_SHOT_RESERVED, f"few_shot_n should be smaller than {FEW_SHOT_RESERVED}"
        self.few_shot_n = few_shot_n

        self.instances = self.read(file_path)
    
    def get_example_dict(self, example: Dict[str, Any], context: str, code: str = "", 
                         train_mode: bool = True) -> Dict[str, Any]:
        example_dict = {"metadata": example}

        if train_mode:
            tokenizer_outputs = self.tokenizer("\n".join([context, code]))
            context_len = len(self.tokenizer.tokenize(context))
            example_dict["context_mask"] = [0] * context_len + [1] * (len(tokenizer_outputs["input_ids"]) - context_len)
        else:
            tokenizer_outputs = self.tokenizer(context)

        example_dict["input_ids"] = tokenizer_outputs["input_ids"]
        example_dict["attention_mask"] = tokenizer_outputs["attention_mask"]

        if train_mode:
            example_dict["input_ids"] += [self.tokenizer.eos_token_id]
            example_dict["attention_mask"] += [1]
            example_dict["context_mask"] += [1]

        example_dict["metadata"]["pad_token_id"] = self.tokenizer.pad_token_id

        return example_dict

    def get_train_instance(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("the base class should not be used directly")

    def get_test_instance(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("the base class should not be used directly")

    def read(self, file_path: str) -> Iterable[Dict[str, Any]]:
        print("Reading dataset files at %s", file_path)

        all_yield_instances = []

        # load the mathqa dataset with states
        mathqa_json_examples = []
        with open(file_path, 'r') as f:
            if self.mode == "test_few_shot":
                raise NotImplementedError("test_few_shot is not implemented yet")
            else:
                lines = f.readlines()[:self.max_instances]
            for line in lines:
                mathqa_json_examples.append(json.loads(line))

        for exp in mathqa_json_examples:
            if self.mode == "train":
                example_dict = self.get_train_instance(exp)
            elif self.mode == "test":
                example_dict = self.get_test_instance(exp)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            all_yield_instances.append(example_dict)

        logger.info(f"loaded {len(all_yield_instances)} instances")

        return all_yield_instances

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def truncate(self, max_instances):
        truncated_instances = self.instances[max_instances:]
        self.instances = self.instances[:max_instances]
        return truncated_instances

    def extend(self, instances):
        self.instances.extend(instances)

def customized_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_dict = {}

    pad_token_id = examples[0]["metadata"]["pad_token_id"]

    for k in examples[0].keys():
        if k == "metadata":
            result_dict[k] = [ex[k] for ex in examples]
        elif k == "input_ids":
            result_dict[k] = left_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                batch_first=True, padding_value=pad_token_id)
        elif k == "attention_mask":
            result_dict[k] = left_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                batch_first=True, padding_value=0)
        elif k == "state_mask":
            result_dict[k] = left_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                batch_first=True, padding_value=0)
        elif k == "context_mask":
            result_dict[k] = left_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                batch_first=True, padding_value=0)
        elif k == "labels":
            result_dict[k] = left_pad_sequences([torch.tensor(ex[k]) for ex in examples], 
                                batch_first=True, padding_value=pad_token_id)
        else:
            raise ValueError(f"Unknown key {k} in example instance")

    return result_dict

class NL2CodeDataModule(LightningDataModule):
    def __init__(self, 
                transformer_model_name: str,
                batch_size: int = 1, 
                val_batch_size: int = 1,
                few_shot_n: int = 0,
                train_file_path: str = None,
                val_file_path: str = None,
                test_file_path: str = None,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize):
        super().__init__()
        self.transformer_model_name = transformer_model_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.few_shot_n = few_shot_n

        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances

        self.train_data = None
        self.val_data = None

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError("the base class should not be used directly")

    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")

        dtloader = DataLoader(self.train_data, batch_size=self.batch_size, 
                               shuffle=True, drop_last=True, collate_fn=customized_collate_fn)
        return dtloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")

        dtloader = DataLoader(self.val_data, batch_size=self.val_batch_size, 
                               shuffle=False, drop_last=True, collate_fn=customized_collate_fn)
        return dtloader

    def test_dataloader(self):
        raise NotImplementedError
    
    def get_gold_program_func(self, example_dict: Dict[str, Any]):
        raise NotImplementedError
    
    def get_gold_answer_func(self, example_dict: Dict[str, Any]):
        raise NotImplementedError