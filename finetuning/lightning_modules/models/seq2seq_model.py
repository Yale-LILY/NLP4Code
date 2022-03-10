import torch
import json
import os
import math
import torch.nn.functional as F
import pytorch_lightning as pl
import io, tokenize, re
import ast, astunparse

from types import ModuleType
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup


from torchmetrics import Metric, MeanMetric, MetricCollection
from pytorch_lightning import LightningModule

from .seq2seq_model_util import get_model
from execution.execution_evaluation import execution_acc, mathqa_execution
from execution.execution_evaluation import execution_eval_at_k, batch_execution_acc

# from https://stackoverflow.com/questions/1769332/script-to-remove-python-comments-docstrings
def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out

def post_process_code(code, remove_comments=True, remove_extra_lines=False, ast_back_parse=True):
    """ a series of post-processing steps to clean up the code and avoid duplicated code """

    if remove_comments:
        code = remove_comments_and_docstrings(code)
    
    if ast_back_parse:
        code = astunparse.unparse(ast.parse(code))

    if remove_extra_lines:
        # remove the code after "answer" is generated
        result = []
        for line in code.split("\n"):
            result.append(line)
            if line.startswith("answer"):
                break
        code = "\n".join(result)

    code = code.strip()

    return code

class Seq2SeqModel(LightningModule):
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
                 load_ckpt_file: str = None) -> None:
        super().__init__()

        self.max_gen_len = max_gen_len
        self.sampling_temp = sampling_temp
        self.sampling_temp_at_k = sampling_temp_at_k
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps

        self.pass_at_k = pass_at_k
        self.eval_pass_at_k_every_n_epochs = eval_pass_at_k_every_n_epochs
        self.max_generation_batches = max_generation_batches
        self.eval_greedy_search = eval_greedy_search

        # We only instantiate this when we need it.
        self.model, self.tokenizer = get_model(transformer_model_name, gradient_ckpt=gradient_ckpt)

        # save the prediction results for every valiation epoch
        self.predictions: List[Dict[str, Any]] = []

        self.opt_params = optimizer["init_args"]
        self.lrs_params = lr_scheduler
        assert lr_scheduler["name"] in ["linear", "cosine", "constant"], "lr_scheduler must be one of 'linear', 'cosine', 'constant'"
        self.lr_scheduler = lr_scheduler

        # load the state dict from the checkpoint file
        if load_ckpt_file is not None:
            checkpoint = torch.load(load_ckpt_file, map_location=torch.device("cpu"))
            self.load_state_dict(checkpoint["state_dict"])
            print(f"loaded weights from {load_ckpt_file}")

        self.metrics_dict: Dict[str, Metric] = MetricCollection({})

        self.metrics_dict["exec_acc"] = MeanMetric()
        self.metrics_dict["exec_rate"] = MeanMetric()
        self.metrics_dict["program_len_diff"] = MeanMetric()

        if self.pass_at_k > 1:
            self.metrics_dict[f"acc@{self.pass_at_k}"]= MeanMetric()
            self.metrics_dict[f"pass@{self.pass_at_k}"]= MeanMetric()

        if self.eval_greedy_search:
            self.metrics_dict["greedy_exec_acc"]= MeanMetric()
            self.metrics_dict["greedy_exec_rate"]= MeanMetric()

    def forward(  # type: ignore
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        The inference time behavior of the model.

        Args:
            input_ids [torch.Tensor]: Tokens from the context. 
            metadata (Optional[List[Dict[str, Any]]], optional): All additional information, `List` for the batch. Defaults to None.

        Returns:
            Dict[str, Any]: results saved in a `Dict` object.
        """        

        generated_token_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, 
                                                max_length=input_ids.shape[1]+self.max_gen_len, 
                                                temperature=self.sampling_temp)

        generated_token_ids = generated_token_ids[:, input_ids.shape[1]:]

        generated_strs = self.tokenizer.batch_decode(generated_token_ids)

        # truncate after the first '#' to be consistent with the codex prompting experiments
        generated_programs = [s.split(self.tokenizer.eos_token)[0] for s in generated_strs]

        output_dicts = [{"generated_program": generated_programs[i], "metadata": metadata[i]} \
                        for i in range(len(generated_programs))]

        if self.eval_greedy_search:
            generated_token_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False, 
                                                    max_length=input_ids.shape[1]+self.max_gen_len)
            generated_token_ids = generated_token_ids[:, input_ids.shape[1]:]
            generated_strs = self.tokenizer.batch_decode(generated_token_ids)
            # truncate after the first '#' to be consistent with the codex prompting experiments
            generated_programs = [s.split(self.tokenizer.eos_token)[0] for s in generated_strs]

            for i in range(len(metadata)):
                output_dicts[i]["greedy_generated_program"] =  generated_programs[i]

        # evaluate pass at k FIXME: a lot of overlapping code here
        if self.current_epoch % self.eval_pass_at_k_every_n_epochs == 0 and self.pass_at_k > 1:
            generated_strs_list = [[] for _ in range(len(metadata))]
            remaining_k = self.pass_at_k
            while remaining_k > 0:
                generate_batch_size = min(remaining_k, self.max_generation_batches)
                remaining_k -= generate_batch_size
                batch_generated_token_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                                        do_sample=True, 
                                                        max_length=input_ids.shape[1]+self.max_gen_len, 
                                                        temperature=self.sampling_temp_at_k, 
                                                        num_return_sequences=generate_batch_size)

                batch_generated_token_ids = batch_generated_token_ids[:, input_ids.shape[1]:]
                batch_generated_strs = self.tokenizer.batch_decode(batch_generated_token_ids)
                batch_generated_programs = [s.split(self.tokenizer.eos_token)[0] for s in batch_generated_strs]

                for i in range(len(metadata)):
                    generated_strs_list[i].extend(batch_generated_programs[i*generate_batch_size:(i+1)*generate_batch_size])

            for i in range(len(metadata)):
                output_dicts[i]["generated_k_programs"] =  generated_strs_list[i]


        return output_dicts

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"] if "labels" in batch else input_ids

        model_result = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.log("loss", model_result.loss, on_step=True, on_epoch=True)
        return {"loss": model_result.loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        # input_tokens, target_mask, context_tokens, target_tokens, metadata = batch
        return self.forward(batch["input_ids"], batch["attention_mask"], batch["metadata"])

    def validation_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        # update the evaluation metrics
        for output_dict in outputs:
            exec_acc = execution_acc(output_dict["generated_program"], mathqa_execution, output_dict["metadata"]["answer"])
            program_len = len(list(filter(lambda x: not x.startswith("#") and not len(x.strip()) == 0, 
                                                output_dict["generated_program"].split("\n"))))
            gold_program_len = len(list(filter(lambda x: not x.startswith("#") and not len(x.strip()) == 0, 
                                                post_process_code(output_dict["metadata"]["code"]).split("\n"))))

            program_len_diff = program_len - gold_program_len

            self.metrics_dict["exec_acc"](exec_acc[0])
            self.metrics_dict["exec_rate"](exec_acc[1])
            self.metrics_dict["program_len_diff"](program_len_diff)

            # also save the results in the json output file
            output_dict["metrics"] = {"exec_acc": float(exec_acc[0]), 
                                      "exec_rate": float(exec_acc[1]),
                                      "program_len_diff": float(program_len_diff)}

            if self.eval_greedy_search:
                exec_acc = execution_acc(output_dict["greedy_generated_program"], mathqa_execution, 
                                         output_dict["metadata"]["answer"])

                self.metrics_dict["greedy_exec_acc"](exec_acc[0])
                self.metrics_dict["greedy_exec_rate"](exec_acc[1])

                output_dict["metrics"].update({"greedy_exec_acc": float(exec_acc[0]), 
                                        "greedy_exec_rate": float(exec_acc[1])})

            # canonocalization of the states to avoid error on saving the modules
            if "generated_program_state_list" in output_dict:
                for state_dict in output_dict["generated_program_state_list"]:
                    if state_dict is not None:
                        for key, value in state_dict.items():
                            if isinstance(value, ModuleType):
                                state_dict[key] = str(value)

        # save the outputs to the model
        self.predictions.extend(outputs)

    def validation_epoch_end_extra(self, outputs: List[Dict[str, Any]]) -> None:
        # compute the eval_at_k metrics
        if self.current_epoch % self.eval_pass_at_k_every_n_epochs == 0 and self.pass_at_k > 1:
            print("evaluating pass at k...")

            all_generated_k_programs = [p["generated_k_programs"] for p in self.predictions]
            all_generated_k_programs_faltten = [item for sublist in all_generated_k_programs for item in sublist]
            gold_answers = [p["metadata"]["answer"] for p in self.predictions]

            result_list = batch_execution_acc(all_generated_k_programs_faltten, mathqa_execution, gold_answers, 
                                           len(self.predictions), self.pass_at_k)
            
            for acc_at_k, pass_at_k in result_list:
                self.metrics_dict[f"acc@{self.pass_at_k}"](acc_at_k)
                self.metrics_dict[f"pass@{self.pass_at_k}"](pass_at_k)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        # extra steps for using the predictions
        self.validation_epoch_end_extra(outputs)

        # compute the metrics
        eval_metrics_dict = {}
        for k in self.metrics_dict.keys():
            if k.startswith("dedup_"):
                dedup_exec_acc = float(self.metrics_dict[k].compute())
                # for the dedup metrics, it's possible that a batch is all duplicates thus manually set nan to 0.0
                eval_metrics_dict[k] = dedup_exec_acc if not math.isnan(dedup_exec_acc) else 0.0
            else:
                eval_metrics_dict[k] = float(self.metrics_dict[k].compute())
        
        # log and save the evalution metrics
        print(f"validation result: {eval_metrics_dict}")
        self.log_dict(eval_metrics_dict) 

        # reset all the metrics
        for k in self.metrics_dict.keys():
            self.metrics_dict[k].reset()

        # save the predictions
        save_pred_file_path = os.path.join(self.trainer.log_dir,
                                f'predictions_step_{self.trainer.global_step}_rank_{self.trainer.global_rank}.jsonl')
        with open(save_pred_file_path, 'w+') as f:
            for prediction in self.predictions:
                f.write(json.dumps(prediction)+'\n')
        print(f"{len(self.predictions)} predictions saved to {save_pred_file_path}")

        # reset the predictions
        self.predictions = []
        
        # NOTE: debug setting only
        # self.sampling_temp += 0.1
        # self.sampling_temp_at_k += 0.2
        # print(f"sampling temp is now {self.sampling_temp}, sampling temp at k is now {self.sampling_temp_at_k}")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.opt_params)
        if self.lr_scheduler["name"] == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **self.lr_scheduler["init_args"])
        elif self.lr_scheduler["name"] == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, **self.lr_scheduler["init_args"])
        elif self.lr_scheduler["name"] == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, **self.lr_scheduler["init_args"])
        else:
            raise ValueError(f"lr_scheduler {self.lr_scheduler} is not supported")

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step"
                    }
                }