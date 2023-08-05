import torch


from itertools import chain
from overrides import overrides
from typing import Dict, Iterable, List, Any, Optional, Union
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from finetuning.lightning_modules.datasets.spider_reader import FewShotSpiderDataset, SpiderDataset
from finetuning.lightning_modules.datasets.mathqa_reader import FewShotMathQADataset, MathQADataset
from finetuning.lightning_modules.datasets.mbpp_reader import FewShotMBPPDataset
from finetuning.lightning_modules.datasets.humaneval_reader import FewShotHumanEvalDataset

from finetuning.lightning_modules.models.seq2seq_model_util import is_model_gpt_style
from finetuning.lightning_modules.models.seq2seq_model_util import left_pad_sequences, right_pad_sequences

def customized_collate_fn_gpt(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return customized_collate_fn(examples, is_left_pad=True)

def customized_collate_fn_enc_dec(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return customized_collate_fn(examples, is_left_pad=False)

def customized_collate_fn(examples: List[Dict[str, Any]], is_left_pad: bool = True) -> Dict[str, Any]:
    result_dict = {}

    pad_token_id = examples[0]["metadata"]["pad_token_id"]

    pad_func = left_pad_sequences if is_left_pad else right_pad_sequences

    for k in examples[0].keys():
        if k == "metadata":
            result_dict[k] = [ex[k] for ex in examples]
        elif k == "input_ids":
            lists_to_pad = list(chain(*[[torch.tensor(t) for t in ex[k]] for ex in examples])) \
                if isinstance(examples[0][k][0], list) else [torch.tensor(ex[k]) for ex in examples]
            result_dict[k] = pad_func(lists_to_pad, batch_first=True, padding_value=pad_token_id)
        elif k == "attention_mask":
            lists_to_pad = list(chain(*[[torch.tensor(t) for t in ex[k]] for ex in examples])) \
                if isinstance(examples[0][k][0], list) else [torch.tensor(ex[k]) for ex in examples]
            result_dict[k] = pad_func(lists_to_pad, batch_first=True, padding_value=0)
        elif k == "labels":
            lists_to_pad = list(chain(*[[torch.tensor(t) for t in ex[k]] for ex in examples])) \
                if isinstance(examples[0][k][0], list) else [torch.tensor(ex[k]) for ex in examples]
            result_dict[k] = pad_func(lists_to_pad, batch_first=True, padding_value=-100)
        else:
            raise ValueError(f"Unknown key {k} in example instance")

    return result_dict

class NL2CodeDataModule(LightningDataModule):
    def __init__(self, 
                transformer_model_name: str,
                dataset_cls: str,
                batch_size: int = None, 
                val_batch_size: int = None,
                # the following settings will override the default values in the init args of the dataset classes
                train_max_instances: int = None,
                val_max_instances: int = None,
                train_file_path: str = None,
                val_file_path: str = None,
                # the following dictionaries are used as default values, as the settings above will override them
                train_set_init_args: Dict[str, Any] = {},
                val_set_init_args: Dict[str, Any] = {},
                set_common_init_args: Dict[str, Any] = {},
                ):
        super().__init__()
        self.transformer_model_name = transformer_model_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        # init the dataset classes
        self.dataset_cls = eval(dataset_cls)

        # delegate the initialization of the train and val datasets to the dataset classes
        self.train_set_init_args = train_set_init_args
        self.train_set_init_args.update(set_common_init_args)
        self.val_set_init_args = val_set_init_args
        self.val_set_init_args.update(set_common_init_args) 

        if train_max_instances is not None:
            self.train_set_init_args["max_instances"] = train_max_instances
        if val_max_instances is not None:
            self.val_set_init_args["max_instances"] = val_max_instances
        if train_file_path is not None:
            self.train_set_init_args["file_path"] = train_file_path
        if val_file_path is not None:
            self.val_set_init_args["file_path"] = val_file_path

        self.train_data = None
        self.val_data = None

    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            train_data = self.dataset_cls(transformer_model_name=self.transformer_model_name,
                                    mode="train", **self.train_set_init_args)
            self.train_data = train_data

        val_data = self.dataset_cls(transformer_model_name=self.transformer_model_name,
                                 mode="test", **self.val_set_init_args)
        self.val_data = val_data 

    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")
        
        collate_fn = customized_collate_fn_gpt if is_model_gpt_style(self.transformer_model_name) \
                                                else customized_collate_fn_enc_dec

        dtloader = DataLoader(self.train_data, batch_size=self.batch_size, 
                               shuffle=True, drop_last=True, collate_fn=collate_fn)
        return dtloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")

        collate_fn = customized_collate_fn_gpt if is_model_gpt_style(self.transformer_model_name) \
                                                else customized_collate_fn_enc_dec

        dtloader = DataLoader(self.val_data, batch_size=self.val_batch_size, 
                               shuffle=False, drop_last=True, collate_fn=collate_fn)
        return dtloader

    def test_dataloader(self):
        raise NotImplementedError

class FewShotNL2CodeDataModule(NL2CodeDataModule):
    def __init__(self, 
                # NOTE: since jsonargparse needs to check all the arguments, we have to repeat the arguments here
                transformer_model_name: str,
                dataset_cls: str,
                batch_size: int = None, 
                val_batch_size: int = None,
                # the following settings will override the default values in the init args of the dataset classes
                train_max_instances: int = None,
                val_max_instances: int = None,
                train_file_path: str = None,
                val_file_path: str = None,
                # the following dictionaries are used as default values, as the settings above will override them
                train_set_init_args: Dict[str, Any] = {},
                val_set_init_args: Dict[str, Any] = {},
                set_common_init_args: Dict[str, Any] = {},
                # following is the few-shot specific settings, but the same overriden rule applies
                prompting_init_args: Dict[str, Any] = {},
                exemplar_file_path: str = None,
                num_exemplars: int = None,
                fixed_exemplars: bool = None,
                exemplar_selection_method: str = None,
                add_instruction: bool = None,
                use_chat_format: bool = None,
                additional_prompt_func_args: Dict[str, Any] = {},
                ):
        # setting and overriding the default values for prompting settings
        self.few_shot_init_args = prompting_init_args
        if exemplar_file_path is not None:
            self.few_shot_init_args["exemplar_file_path"] = exemplar_file_path
        if num_exemplars is not None:
            self.few_shot_init_args["num_exemplars"] = num_exemplars
        if fixed_exemplars is not None:
            self.few_shot_init_args["fixed_exemplars"] = fixed_exemplars
        if exemplar_selection_method is not None:
            self.few_shot_init_args["exemplar_selection_method"] = exemplar_selection_method
        if add_instruction is not None:
            self.few_shot_init_args["add_instruction"] = add_instruction
        if use_chat_format is not None:
            self.few_shot_init_args["use_chat_format"] = use_chat_format
        
        self.additional_prompt_func_args = additional_prompt_func_args

        super().__init__(transformer_model_name=transformer_model_name,
                         dataset_cls=dataset_cls,
                         batch_size=batch_size,
                         val_batch_size=val_batch_size,
                         train_max_instances=train_max_instances,
                         val_max_instances=val_max_instances,
                         train_file_path=train_file_path,
                         val_file_path=val_file_path,
                         train_set_init_args=train_set_init_args,
                         val_set_init_args=val_set_init_args,
                         set_common_init_args=set_common_init_args)
    @overrides
    def train_dataloader(self):
        raise NotImplementedError("train_dataloader is not implemented for FewShotNL2CodeDataModule")

    @overrides
    def setup(self, stage: Optional[str] = None):
        assert stage in ["validate"], "FewShotNL2CodeDataModule only supports validate stage"

        val_data = self.dataset_cls(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args, 
                                    **self.few_shot_init_args, additional_prompt_func_args=self.additional_prompt_func_args)

        self.val_data = val_data 