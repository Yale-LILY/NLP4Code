from typing import Dict, Iterable, List, Any, Optional, Union, Tuple
from overrides import overrides

from finetuning.lightning_modules.datasets.base_reader import NL2CodeDataset, FewShotNL2CodeDataset

class MathQADataset(NL2CodeDataset):

    @overrides
    def get_train_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.get_example_dict(example, example["text"], example["code"], train_mode=True)]

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        # parse the answer and add the field
        example["original_answer"] = example["answer"]
        # TODO: in data/mathqa/val_dedup.jsonl, example["answer"] are floats
        # example["answer"] = example["answer"].split("\n####")[-1].strip()

        return [self.get_example_dict(example, example["text"], "", train_mode=False)]

class FewShotMathQADataset(FewShotNL2CodeDataset):

    instruction: str = "## Given questions in the comment, use python programs to produce the correct answers with the `answer` variable."

    @overrides
    def get_test_instance(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = self.get_prompt_for_example(example)

        if not (isinstance(example["answer"], float) or isinstance(example["answer"], int)):
            # this is gsm8k and not mathqa so parse the answer and add the field
            example["original_answer"] = example["answer"]
            try:
                example["answer"] = float(example["answer"].split("\n####")[-1].strip().replace(",", ""))
            except Exception as e:
                example["answer"] = -100000.0 # FIXME: this is a hack to make sure the answer is wrong

        return [self.get_example_dict(example, context, train_mode=False)]
    
    # @overrides
    def promptify_example(self, example: Dict[str, Any], add_code: bool = True) -> Tuple[str, str]:
        if add_code:
            assert "annotated_code" in example, "annotated_code not found in exemplar dict"
        
        nl_input = "## " + example["question"]
        code = example["annotated_code"] if add_code else ""

        return nl_input, code