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


n_examples = 50
mode = "test_few_shot"
few_shot_n = 10
k = 80

mathqa_dataset = MathQADataset(
    "/data/lily/nos6/NLP4Code/NLP4Code/data/mathqa/val-python.jsonl",
    "EleutherAI/gpt-neo-125M",
    n_examples,
    mode=mode,
    few_shot_n=few_shot_n,
)
print("Dataset loaded.\n")



prompt = "# Answer the following math word problem in Python.\n"


print("Evaluating...\n")
programs, acc, pass_at_k = codex_evaluate_pass_at_k(
    mathqa_dataset,
    prompt,
    mathqa_ins_text,
    mathqa_ins_answer,
    mathqa_execution,
    get_ins_few_shot_text=mathqa_ins_few_shot_text,
    get_ins_few_shot_soln=mathqa_ins_few_shot_soln,
    eval_at_k=k,
    few_shot_enabled=True,
)

print("Evaluated on {} examples...\n".format(n_examples))
print("Used {} examples for few-shot evaluation.\n".format(few_shot_n))
print(k = 80)
print("Used prompt: \nprompt")
print(f"Pass@{k}:", pass_at_k)
print("Accuracy:", acc)
