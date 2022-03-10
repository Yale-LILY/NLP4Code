from unicodedata import name
import torch
import json
import os
import math
import torch.nn.functional as F
import multiprocessing
import io, tokenize, re
import ast, astunparse

from types import ModuleType
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration

from torchmetrics import Metric, MeanMetric, MetricCollection
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from .codet5_util import get_codet5
from .gpt_util import left_pad_sequences
from .gpt_seq2seq_model import *
from execution.execution_evaluation import execution_acc, mathqa_execution
from execution.execution_evaluation import execution_eval_at_k, batch_execution_acc

class CodeT5Seq2SeqModel(GptSeq2SeqModel):
    def __init__(self, 
                 transformer_model_name: str,
                 max_gen_len: int = 100,
                 sampling_temp: float = 0.2,
                 sampling_temp_at_k: float = 0.8,
                 gradient_ckpt: bool = False,
                 pass_at_k: int = 1,
                 eval_pass_at_k_every_n_epochs: int = 1,
                 max_generation_batches: int = 100,
                 max_steps: int = -1,
                 warmup_steps: int = 0,
                 eval_greedy_search: bool = False,
                 optimizer: Dict[str, Any] = None,
                 lr_scheduler: Dict[str, Any] = None,
                 load_ckpt_file: str = None,
                 get_model_fn = get_codet5,
                 ) -> None:
        
        # We only instantiate this when we need it.
        super(CodeT5Seq2SeqModel, self).__init__(
            transformer_model_name = transformer_model_name,
            max_gen_len = max_gen_len,
            sampling_temp = sampling_temp,
            sampling_temp_at_k = sampling_temp_at_k,
            gradient_ckpt = gradient_ckpt,
            pass_at_k = pass_at_k,
            eval_pass_at_k_every_n_epochs = eval_pass_at_k_every_n_epochs,
            max_generation_batches = max_generation_batches,
            max_steps = max_steps,
            warmup_steps = warmup_steps,
            eval_greedy_search = eval_greedy_search,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            load_ckpt_file = load_ckpt_file,
            get_model_fn = get_model_fn,
        )
            
    def forward(  # type: ignore
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        return super(CodeT5Seq2SeqModel, self).forward(input_ids, attention_mask, metadata)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return super(CodeT5Seq2SeqModel, self).training_step(batch, batch_idx)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        return super(CodeT5Seq2SeqModel, self).validation_step(batch, batch_idx)

    def validation_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        super(CodeT5Seq2SeqModel, self).validation_step_end(outputs)

    def validation_epoch_end_extra(self, outputs: List[Dict[str, Any]]) -> None:
        super(CodeT5Seq2SeqModel, self).validation_epoch_end_extra(outputs)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        super(CodeT5Seq2SeqModel, self).validation_epoch_end(outputs)

    # def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
    #     raise NotImplementedError
    
    # deprecated
    def _configure_optimizers(self):
        return super(CodeT5Seq2SeqModel, self)._configure_optimizers()