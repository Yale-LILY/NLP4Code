from typing import Dict, Iterable, List, Any, Optional, Union, Tuple
from overrides import overrides

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, FewShotNL2CodeDataset

def state_simple_str_func(state_dict: Dict[str, Any]):
    if isinstance(state_dict, str):
        return state_dict
    elif state_dict is not None and 'answer' in state_dict:
        return str(state_dict)
    else:
        return "ERROR"

def state_answer_only_func(state_dict: Dict[str, Any]):
    if isinstance(state_dict, str):
        return state_dict
    elif state_dict is not None and 'answer' in state_dict:
        return str(state_dict['answer'])
    else:
        return "ERROR"

def saved_promptify_mathqa(prompt_file_path: str, example: Dict, max_prompt_examples: int = 100) -> str:
    with open(prompt_file_path, 'r') as f:
        prompt = f.read()

    prompt += "\n\n## " + example["question"]

    return prompt

class MathQADataset(NL2CodeDataset):

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.get_example_dict(example, example["text"], example["code"], train_mode=True)]

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        # parse the answer and add the field
        example["original_answer"] = example["answer"]
        example["answer"] = example["answer"].split("\n####")[-1].strip()

        return [self.get_example_dict(example, example["text"], "", train_mode=False)]

class FewShotMathQADataset(FewShotNL2CodeDataset):

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = self.get_prompt_for_example(example, add_code=False)

        if not (isinstance(example["answer"], float) or isinstance(example["answer"], int)):
            # this is gsm8k and not mathqa so parse the answer and add the field
            example["original_answer"] = example["answer"]
            try:
                example["answer"] = float(example["answer"].split("\n####")[-1].strip().replace(",", ""))
            except Exception as e:
                example["answer"] = -100000.0

        return [self.get_example_dict(example, context, train_mode=False)]
    
    def promptify_mathqa(self, example: Dict[str, Any], add_code: bool = True) -> Tuple[str, str]:
        if add_code:
            assert "annotated_code" in example, "annotated_code not found in exemplar dict"
        
        nl_input = "## " + example["question"]
        code = example["annotated_code"] if add_code else ""

        return nl_input, code