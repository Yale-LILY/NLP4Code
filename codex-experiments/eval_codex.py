from torch import true_divide
from tqdm import tqdm
import time

import openai
from torch.utils.data import Dataset

from evaluation.codex import codex
from finetuning.lightning_modules.datasets.mathqa_reader import MathQADataset
from execution.execution_evaluation import batch_execution_acc, execution_eval_at_k, mathqa_execution


print("Loading dataset...")
mathqa_dataset = MathQADataset("/data/lily/nos6/NLP4Code/NLP4Code/data/mathqa/train-python.jsonl", "EleutherAI/gpt-neo-125M", 2, mode="test_few_shot", few_shot_n=10)

prompt = "# Answer the following math question:"

def codex_evaluate_pass_at_k(input_dataset: Dataset, prompt: str, eval_at_k: int=1, few_shot_enabled: bool=False):
    programs = []
    running_acc, running_pass_at_k = 0, 0
    for item in tqdm(input_dataset.instances):
        
        # prepare input for few-shot experiments if applicable
        if few_shot_enabled is True:
            processed_input = create_few_shot_prompt(prompt, item)

            processed_input = processed_input + "\n\n" + "\n".join((prompt, item["metadata"]["text"]))
        else:
            processed_input = "\n\n".join((prompt, item["metadata"]["text"]))
        # print(processed_input)

        # perform generation with codex
        while True:
            try:
                result = codex([processed_input], engine="code-davinci-001", temperature=0.8, top_p=1, max_tokens=128)
                break
            except openai.error.RateLimitError as e:
                print("RateLimitError occurred, waiting for 60 seconds and retrying...")
                time.sleep(60)
        
        print(result[0])
        program = "\n".join((item["metadata"]["text"], result[0]))

        # check execution accuracy and pass at k for the instance
        acc, pass_k = execution_eval_at_k(program, mathqa_execution, item["metadata"]["answer"], eval_at_k)
        print(acc, int(pass_k))
        running_acc += acc
        running_pass_at_k += pass_k

        programs.append(program)

        
    avg_acc = running_acc / len(input_dataset.instances)
    avg_pass_at_k = running_pass_at_k / len(input_dataset.instances)   
        
    return programs, avg_acc, avg_pass_at_k
        
def create_few_shot_prompt(prompt, item):
    """Appends the prompt to each few shot question and its solution, and concatenates these examples"""
    processed_input = ""
    for text, code in zip(item["metadata"]["few_shot_text"], item["metadata"]["few_shot_code"]):

        if processed_input != "":

            processed_input = processed_input + "\n\n"

        processed_input = processed_input + "\n".join((prompt, text, code))

    # processed_input = processed_input + "\n\n"

    return processed_input





print("Evaluating...")
programs, acc, pass_at_k = codex_evaluate_pass_at_k(mathqa_dataset, prompt, eval_at_k=1, few_shot_enabled=True)

print("Pass@1:", pass_at_k)
print("Accuracy:", acc)

# print(programs)