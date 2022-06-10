import json
import random

random.seed(333)

from finetuning.lightning_modules.datasets.mathqa_reader import MathQADataset

from execution.mathqa_execution import mathqa_execution, mathqa_answer_eq
from codex import codex_evaluate_pass_at_k

print("Loading datasets...\n")
# with open("data/mathqa/train_dedup.jsonl", "r") as f:
with open("data/gsmath/gsmath_train.jsonl", "r") as f:
    training_data = [json.loads(line) for line in f.readlines()]
    random.shuffle(training_data)

# with open("data/mathqa/val_dedup.jsonl", "r") as f:
with open("data/gsmath/gsmath_val.jsonl", "r") as f:
    val_data = [json.loads(line) for line in f.readlines()]
    random.shuffle(val_data)

n_examples = 100
few_shot_n = 4
ks = [5, 10, 20, 50, 100]
# ks = [1]

few_shot_instances = training_data[:few_shot_n]
test_instances = val_data[:n_examples]

print(f"Evaluating {few_shot_n}-shot pass@{str(ks)} performance on {n_examples} examples...")

codex_evaluate_pass_at_k(few_shot_examples=few_shot_instances,
                         input_dataset=test_instances,
                         text_header=MathQADataset.task_prompt_header,
                         promptify_func=MathQADataset.promptify,
                         exec_func=mathqa_execution,
                         answer_eq_func=mathqa_answer_eq,
                         eval_at_ks=ks,
                         openai_kwargs={"temperature": 0.8},
                         save_result_path=f"gsmath_codex_{few_shot_n}_shot_pass_at_{str(ks)}results_{n_examples}.jsonl")

