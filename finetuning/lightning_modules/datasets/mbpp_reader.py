import re
import os
import pandas as pd

from overrides import overrides

from typing import Dict, Iterable, List, Any, Optional, Union, Tuple

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, NL2CodeDataModule
from execution.program_tracing import assertion_to_test

"""
The structure of an example of MBPP:
{
    'task_id': 1,
    'text': 'Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].',
    'code': 'R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]',
    'test_list': [
        'assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8',
        'assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12',
        'assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16'],
    'test_setup_code': '',
    'challenge_test_list': []
}

"""

def mbpp_example_to_demonstration(example: Dict[str, Any], train=True, 
                                  add_assertion_n: int = 0, test_input_only: bool = False) -> str:
    # get the assertions
    if not test_input_only:
        assertion_header = '# These are the assertions for your function:\n'
        for test_case in example['test_list'][:add_assertion_n]:
            assertion_header += test_case + '\n'
    else:
        raise NotImplementedError("test_input_only is not implemented for MBPP")


    # separate the function header and the function body
    func_signature = example["func_signature"]
    func_body = example["func_body"]

    func_comment = f'""" {example["text"]} """'

    header = assertion_header + '\n' + func_comment if add_assertion_n > 0 else func_comment

    if train:
        return f'### Task Start ###\n{header}\n{example["code"]}\n### Task End ###'
    else:
        return f'### Task Start ###\n{header}'

def saved_promptify_mbpp(prompt_file: str, example: Dict[str, Any], add_assertion_n: int) -> str:
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    return prompt + "\n\n" + mbpp_example_to_demonstration(example, train=False, add_assertion_n=add_assertion_n)

def simple_val_str_func(val_name, val_dict) -> str:
    val_type = val_dict["type"].split(" ")[1].replace("'", "").replace(">", "")
    val_val = val_dict["str_value"] if "object" not in val_dict["str_value"] else "<object list>"
    return f"{val_name}({val_type})={val_val}"

def state_simple_str_func(program_exec_dict: Dict[str, Any]) -> str:
    val_strs = []
    if len(program_exec_dict["tracing_local_list"]) == 0:
        return "tracing failed"
    else:
        for k, v in program_exec_dict["tracing_local_list"][0].items():
            if k != "_return_val":
                val_strs.append(simple_val_str_func(k, v))
        
        return ", ".join(val_strs)


class FewShotMBPPDataset(NL2CodeDataset):

    def __init__(self, 
                 prompt_file: str,
                 add_assertion_n: int,
                 **kwargs):
        # init some dataset specific variables
        self.prompt_file = prompt_file
        self.add_assertion_n = add_assertion_n

        super().__init__(**kwargs)

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise ValueError("Few shot datasets do not support training")

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = saved_promptify_mbpp(self.prompt_file, example, self.add_assertion_n)

        return [self.get_example_dict(example, context, train_mode=False)]

class FewShotMBPPDataModule(NL2CodeDataModule):

    @overrides
    def setup(self, stage: Optional[str] = None):
        # OPTIONAL, called for every GPU/machine (assigning state is OK)
        assert stage in ["fit", "validate"]

        if stage == "fit":
            raise ValueError("Few shot datasets do not support training")

        if self.val_data is None:
            val_data = FewShotMBPPDataset(transformer_model_name=self.transformer_model_name,
                                    mode="test", **self.val_set_init_args)
            self.val_data = val_data 