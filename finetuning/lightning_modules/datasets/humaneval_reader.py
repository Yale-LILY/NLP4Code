import re
import os
import pandas as pd

from overrides import overrides

from typing import Dict, Iterable, List, Any, Optional, Union, Tuple

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, FewShotNL2CodeDataset
from execution.program_tracing import assertion_to_test

from human_eval.data import write_jsonl, read_problems


class FewShotHumanEvalDataset(FewShotNL2CodeDataset):

    instruction: str = ""
    example_io_sep: str = "\n"

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = self.get_prompt_for_example(example)

        return [self.get_example_dict(example, context, train_mode=False)]

    # @overrides
    def promptify_example(self, example: Dict[str, Any], add_code: bool = True, 
                          add_assertion_n: int = 0, test_input_only: bool = False) -> Tuple[str, str]:

        header = example["prompt"]

        if add_code:
            return header, f'{example["canonical_solution"]}\n\n'
        else:
            return header, ''