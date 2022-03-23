from finetuning.lightning_modules.datasets.mathqa_reader import MathQADataset

from execution.execution_evaluation import mathqa_execution
from eval_codex import (
    codex_evaluate_pass_at_k,
    mathqa_ins_text,
    mathqa_ins_answer,
    mathqa_ins_few_shot_text,
    mathqa_ins_few_shot_soln,
)

print("Loading dataset...\n")
mathqa_dataset = MathQADataset(
    "/data/lily/nos6/NLP4Code/NLP4Code/data/mathqa/val-python.jsonl",
    "EleutherAI/gpt-neo-125M",
    50,
    mode="test_few_shot",
    few_shot_n=10,
)

prompt = "# Answer the following math question:\n"


print("Evaluating...\n")
programs, acc, pass_at_k = codex_evaluate_pass_at_k(
    mathqa_dataset,
    prompt,
    mathqa_ins_text,
    mathqa_ins_answer,
    mathqa_execution,
    get_ins_few_shot_text=mathqa_ins_few_shot_text,
    get_ins_few_shot_soln=mathqa_ins_few_shot_soln,
    eval_at_k=80,
    few_shot_enabled=True,
)

print("Pass@k:", pass_at_k)
print("Accuracy:", acc)
