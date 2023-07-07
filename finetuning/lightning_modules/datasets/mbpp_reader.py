import re
import os
import pandas as pd

from overrides import overrides

from typing import Dict, Iterable, List, Any, Optional, Union, Tuple

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, FewShotNL2CodeDataset
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

class FewShotMBPPDataset(FewShotNL2CodeDataset):

    instruction: str = "## Given the natural language description and example assertion(s), write a python function."
    example_io_sep: str = "\n"

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = self.get_prompt_for_example(example)

        return [self.get_example_dict(example, context, train_mode=False)]

    # @overrides
    def promptify_example(self, example: Dict[str, Any], add_code: bool = True, 
                          add_assertion_n: int = 0, test_input_only: bool = False) -> Tuple[str, str]:

        # get the assertions
        if not test_input_only:
            assertion_header = '# These are the assertions for your function:\n'
            for test_case in example['test_list'][:add_assertion_n]:
                assertion_header += test_case + '\n'
        else:
            assertion_header = '# These are the calls for your function:\n'
            for test_case in example['test_list'][:add_assertion_n]:
                assertion_header += assertion_to_test(test_case) + '\n'

        # construct the example prompt
        if 'text' in example:
            func_comment = f'""" {example["text"]} """'
        else:
            assert 'prompt' in example, "key 'text' or 'prompt' must be in the example!"
            func_comment = f'""" {example["prompt"]} """'

        header = assertion_header + '\n' + func_comment if add_assertion_n > 0 else func_comment

        if add_code:
            return f'### Task Start ###\n{header}', f'{example["code"]}\n### Task End ###'
        else:
            return f'### Task Start ###\n{header}', ''
