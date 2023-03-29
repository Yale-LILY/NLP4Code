import os
import json
import subprocess

from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor as Pool

from tqdm import tqdm

from execution.program_tracing import get_function_final_state, assertion_to_test
from execution.executors import MBPPExecutor

from finetuning.lightning_modules.datasets.mbpp_reader import mbpp_example_to_demonstration

"""
According to the original repo: https://github.com/google-research/google-research/tree/master/mbpp

We specify a train and test split to use for evaluation. Specifically:

Task IDs 11-510 are used for testing.
Task IDs 1-10 were used for few-shot prompting and not for training.
Task IDs 511-600 were used for validation during fine-tuning.
Task IDs 601-974 are used for training.
"""

def process_python_code(code: str) -> str:
    subprocess.run(["ls", "-l"])

def split_function(code: str, delimiter: str = '\n') -> Tuple[str, str]:
    # separate the function header and the function body
    lines = code.split('\n')

    func_head_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('def'):
            func_head_idx = i
            break
    
    # get the function header and body
    func_signature = '\n'.join(lines[:func_head_idx+1])
    func_body = '\n'.join(lines[func_head_idx+1:])

    return func_signature, func_body

def process_mbpp_example(example: Dict[str, Any]) -> Dict[str, Any]:
    example['code'] = example['code'].replace('\r\n', '\n')
    example["func_signature"], example["func_body"] = split_function(example["code"])

    return None

def data_split(file_name: str):
    with open(file_name, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    for example in data:
        process_mbpp_example(example)
    
    assert len(data) == 974

    # split all the data
    prompt_data = data[:10]
    test_data = data[10:510]
    dev_data = data[510:600]
    train_data = data[600:]

    # save the data
    with open('data/mbpp/mbpp_prompt.jsonl', 'w+') as f:
        for example in prompt_data:
            f.write(json.dumps(example) + '\n')

    with open('data/mbpp/mbpp_test.jsonl', 'w+') as f:
        for example in test_data:
            f.write(json.dumps(example) + '\n')
    
    with open('data/mbpp/mbpp_dev.jsonl', 'w+') as f:
        for example in dev_data:
            f.write(json.dumps(example) + '\n')

    with open('data/mbpp/mbpp_train.jsonl', 'w+') as f:
        for example in train_data:
            f.write(json.dumps(example) + '\n')

def create_mbpp_prompt(prompt_examples_path: str, save_file_path: str, add_assertion_n: int, test_input_only: bool):
    with open(prompt_examples_path, 'r') as f:
        mbpp_prompt_examples = [json.loads(line) for line in f.readlines()]
    
    mbpp_prompt_examples = mbpp_prompt_examples[1:4] # according to the original repo, we use the task id 2, 3, 4

    if add_assertion_n > 0:
        task_description = "# Write Python function to complete the task and pass the assertion tests. \n\n"
    else:
        task_description = "# Write Python function to complete the tasks. \n\n"
    
    example_str_list = []
    for prompt_example in mbpp_prompt_examples:
        example_str = mbpp_example_to_demonstration(prompt_example, train=True, add_assertion_n=add_assertion_n, test_input_only=test_input_only)
        example_str_list.append(example_str)
    
    final_prompt =  task_description + "\n\n".join(example_str_list)

    with open(save_file_path, 'w+') as f:
        f.write(final_prompt)

def get_gen_logprob(tokens: List[str], token_logprobs: List[float], end_seq: str = "### Task End ###") -> float:
    end_token_idx = -1
    for i in range(len(tokens)):
        if tokens[i].startswith(end_seq[0]) and "".join(tokens[i:]).startswith(end_seq):
            end_token_idx = i
            break
    
    program_logprob = sum(token_logprobs[:end_token_idx])
    return program_logprob, program_logprob / end_token_idx

def create_verification_data(codex_output_file: str, dataset_output_file: str):
    with open(codex_output_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    # instantiate a mbpp executor
    mbpp_executor = MBPPExecutor(n_processes=100)
    
    processed_examples = []
    for example in tqdm(data):
        processed_example = {"metadata": example["metadata"]}
        generated_programs = [x['program'].strip() for x in example["generated_k_programs"]]

        # add the gold program, and the generated one program for convenience
        generated_programs = [example["metadata"]["code"].strip()] + \
                             [example["generated_program"]["program"].strip()] + generated_programs

        # then for all the tests, get the program correctness for the generated k programs
        batch_exec_results = mbpp_executor.batch_exec_programs(generated_programs, [example["metadata"]], len(generated_programs))

        # now separate the gold program, generated_program, and the generated k programs
        processed_example["gold_program"] = {"program": generated_programs[0], 
                                             "exec_match": batch_exec_results[0][0],
                                             "exec_states": batch_exec_results[0][1]}

        processed_example["generated_program"] = {"program": generated_programs[1], 
                                                  "exec_match": batch_exec_results[1][0],
                                                  "exec_states": batch_exec_results[1][1]}

        processed_example["generated_k_programs"] = [{"program": generated_programs[i], 
                                                    "exec_match": batch_exec_results[i][0],
                                                    "exec_states": batch_exec_results[i][1],
                                                    } for i in range(2, len(generated_programs))]
        
        for i, program_dict in enumerate(processed_example["generated_k_programs"]):
            program_logprobs, norm_program_logprobs = get_gen_logprob(example["generation_tokens"][i], example["generation_probs"][i])
            program_dict.update({"gen_prob": program_logprobs, "norm_gen_prob": norm_program_logprobs})
        
        processed_examples.append(processed_example)

    with open(dataset_output_file, 'w+') as f:
        for example in processed_examples:
            f.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    # data_split("data/mbpp/mbpp.jsonl")

    # create_mbpp_prompt('data/mbpp/mbpp_prompt.jsonl', 'prompt_files/mbpp_prompt_3_test.txt', 
    #                    add_assertion_n=3, test_input_only=True)

    create_verification_data("results/mbpp-codex_davinci-few_shot-train-pass_at_100-1_test_input_only/predictions_step_0_rank_0.jsonl", 
                             'data/mbpp/mbpp_input_only_verification_train.jsonl')
