import json
import logging
import sys
import os
import torch
import random

from typing import Dict, Iterable, List, Any, Optional, Union, Tuple
from itertools import chain
from overrides import overrides
from tqdm import tqdm

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from .reader_utils import CHAT_SEP_TOKEN
from finetuning.lightning_modules.models.seq2seq_model_util import is_model_gpt_style, right_pad_sequences
from finetuning.lightning_modules.models.seq2seq_model_util import get_model, left_pad_sequences

# set environment variable to avoid deadlocks, see: 
# https://docs.allennlp.org/main/api/data/data_loaders/multiprocess_data_loader/#multiprocessdataloader.common_issues
os.environ['TOKENIZERS_PARALLELISM']='0'


logger = logging.getLogger(__name__)

class NL2CodeDataset(Dataset):

    def __init__(
        self, 
        file_path: str,
        transformer_model_name: str, 
        max_instances: int = sys.maxsize,
        mode: str = "train", 
        mask_context_loss: bool = False,
        multi_instance_example: bool = False,
        enable_tqdm: bool = False,
        generation_length: int = 128,
        stats_keys: List[str] = ["total_instances", "input_too_long"],
        **kwargs):
        super().__init__(**kwargs)

        # mode is one of ["train", "test", "test_few_shot"]
        assert mode in ["train", "test", "test_few_shot"]

        self.transformer_model_name = transformer_model_name
        _, self.tokenizer = get_model(transformer_model_name, tokenizer_only=True)

        self.mask_context_loss = mask_context_loss
        assert not self.mask_context_loss or is_model_gpt_style(self.transformer_model_name), \
            "mask_context_loss is only supported for GPT-style models"

        self.max_instances = max_instances
        self.mode = mode
        self.multi_instance_example = multi_instance_example
        self.enable_tqdm = enable_tqdm
        self.generation_length = generation_length

        # use to report dataset statistics
        self.stats = dict()
        for key in stats_keys:
            self.stats[key] = 0

        self.instances = self.read(file_path)

    def get_example_dict_gpt(self, example: Dict[str, Any], context: str, code: str = "", 
                         train_mode: bool = True, length_cutoff: bool = True) -> Dict[str, Any]:
        example_dict = {"metadata": example}

        if train_mode:
            tokenizer_outputs = self.tokenizer("\n".join([context, code]), truncation=length_cutoff)
            context_len = len(self.tokenizer(context + "\n", truncation=length_cutoff)["input_ids"])
            if self.mask_context_loss:
                example_dict["labels"] = [-100] * context_len + tokenizer_outputs["input_ids"][context_len:]
            else:
                example_dict["labels"] = tokenizer_outputs["input_ids"].copy()
        else:
            tokenizer_outputs = self.tokenizer(context + "\n", truncation=length_cutoff)

        example_dict["input_ids"] = tokenizer_outputs["input_ids"]
        example_dict["attention_mask"] = tokenizer_outputs["attention_mask"]

        if train_mode:
            example_dict["input_ids"] += [self.tokenizer.eos_token_id]
            example_dict["labels"] += [self.tokenizer.eos_token_id]
            example_dict["attention_mask"] += [1]

        example_dict["metadata"]["pad_token_id"] = self.tokenizer.pad_token_id

        return example_dict
    
    def get_example_dict_enc_dec(self, example: Dict[str, Any], context: str, code: str = "", 
                         train_mode: bool = True, length_cutoff: bool = True) -> Dict[str, Any]:
        example_dict = {"metadata": example}

        context_tokenizer_outputs = self.tokenizer(context, truncation=length_cutoff)
        example_dict["input_ids"] = context_tokenizer_outputs["input_ids"]
        example_dict["attention_mask"] = context_tokenizer_outputs["attention_mask"]

        if train_mode:
            code_tokenizer_outputs = self.tokenizer(code, truncation=length_cutoff)
            example_dict["labels"] = code_tokenizer_outputs["input_ids"]

        example_dict["metadata"]["pad_token_id"] = self.tokenizer.pad_token_id

        return example_dict
    
    def get_example_dict(self, example: Dict[str, Any], context: str, code: str = "", 
                         train_mode: bool = True, length_cutoff: bool = True) -> Dict[str, Any]:
        if not is_model_gpt_style(self.transformer_model_name):
            example_dict = self.get_example_dict_enc_dec(example, context, code, train_mode, length_cutoff=length_cutoff)
        else:
            example_dict = self.get_example_dict_gpt(example, context, code, train_mode, length_cutoff=length_cutoff)
        
        if len(example_dict["input_ids"]) + self.generation_length > self.tokenizer.model_max_length:
            self.stats["input_too_long"] += 1
        
        return example_dict

    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("the base class should not be used directly")

    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
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

        iters = tqdm(mathqa_json_examples) if self.enable_tqdm else mathqa_json_examples
        for exp in iters:
            if self.mode == "train":
                example_dict = self.get_train_instance(exp)
            elif self.mode == "test":
                example_dict = self.get_test_instance(exp)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # note that the returned example_dict might be a list of dicts
            all_yield_instances.extend(example_dict)

        logger.info(f"loaded {len(all_yield_instances)} instances")

        self.stats["total_instances"] = len(all_yield_instances)
        self.report_statistics()

        return all_yield_instances
    
    def report_statistics(self):
        total = self.stats["total_instances"]

        dataset_stats = "-" * 30 + "\nDataset statistics:\n"
        for key, value in self.stats.items():
            if key == "total_instances":
                continue
            dataset_stats += f"{key}: {value/total:.1%} \n"  
        dataset_stats += "-" * 30
        print(dataset_stats)

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

class FewShotNL2CodeDataset(NL2CodeDataset):

    # class variables, can be overwritten by subclasses and can be changed in the init function
    instruction: str = None
    example_io_sep: str = ""
    between_example_sep: str = "\n\n"

    def __init__(
        self, 
        mode: str = "test", 
        # exemplar settings, these settings are also overridable from the command line as 
        # they are also arguements for the datamodule class
        exemplar_file_path: str = None,
        num_exemplars: int = None,
        fixed_exemplars: bool = True,
        exemplar_selection_method: str = "first",
        add_instruction: bool = True,
        use_chat_format: bool = False,
        additional_prompt_func_args: Dict[str, Any] = {},
        # override class variables
        instruction: str = None,
        example_io_sep: str = None,
        between_example_sep: str = None,
        **kwargs):

        assert mode == "test", "FewShotNL2CodeDataset only supports test mode"
        assert exemplar_selection_method in ["first", "random"], "exemplar_selection_method must be first or random"

        self.exemplar_file_path = exemplar_file_path
        self.num_exemplars = num_exemplars
        self.fixed_exemplars = fixed_exemplars
        self.exemplar_selection_method = exemplar_selection_method

        self.add_instruction = add_instruction
        self.use_chat_format = use_chat_format
        self.additional_prompt_args = additional_prompt_func_args

        if instruction is not None:
            self.instruction = instruction
        if example_io_sep is not None:
            self.example_io_sep = example_io_sep
        if between_example_sep is not None:
            self.between_example_sep = between_example_sep

        if self.use_chat_format:
            self.example_io_sep = CHAT_SEP_TOKEN + self.example_io_sep 
            self.between_example_sep = CHAT_SEP_TOKEN + self.between_example_sep

        # read the exemplar file and 
        with open(exemplar_file_path, 'r') as f:
            all_exemplars = [json.loads(s) for s in f.readlines()]
        self.exemplar_nl_code_pairs: List[Tuple[str, str]] = \
            [self.promptify_example(example, add_code=True, **self.additional_prompt_args) for example in all_exemplars]

        if self.fixed_exemplars:
            # we pre-select the exemplars
            if self.exemplar_selection_method == "first":
                self.exemplar_nl_code_pairs = self.exemplar_nl_code_pairs[:self.num_exemplars]
            elif self.exemplar_selection_method == "random":
                random.shuffle(self.exemplar_nl_code_pairs)
                self.exemplar_nl_code_pairs = self.exemplar_nl_code_pairs[:self.num_exemplars]
            else:
                raise ValueError(f"Unknown exemplar_selection_method: {self.exemplar_selection_method}")
        
        super().__init__(mode=mode, **kwargs)

    def get_prompt_for_example(self, example: Dict[str, Any]) -> str:
        """ with the instruction, connect the components of the example, and then connect the examples """
        # promptify the current example
        nl_input, _ = self.promptify_example(example, add_code=False, **self.additional_prompt_args)

        if self.fixed_exemplars or self.exemplar_selection_method == "first":
            example_exemplars = self.exemplar_nl_code_pairs[:self.num_exemplars]
        elif self.exemplar_selection_method == "random":
            random.shuffle(self.exemplar_nl_code_pairs)
            example_exemplars = self.exemplar_nl_code_pairs[:self.num_exemplars]
        else:
            raise ValueError(f"Unknown exemplar_selection_method: {self.exemplar_selection_method}")
        
        # construct the actual prompt
        prompt = self.instruction + self.between_example_sep if self.add_instruction else ""
        for nl, code in example_exemplars:
            prompt += nl + self.example_io_sep + code + self.between_example_sep
        prompt += nl_input # the model needs to learn to generate the separator by itself

        return prompt

    def promptify_example(self, example: Dict[str, Any], add_code: bool = True, **kwargs) -> Tuple[str, str]:
        """ given an example json dict, return the input (program_context, nl) and output (code) """
        raise NotImplementedError("promptify_example must be implemented by the subclass")

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise ValueError("FewShotNL2CodeDataset does not support training")