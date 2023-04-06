import re
import json
import os 
import multiprocessing

import random

from tqdm import tqdm

from typing import List, Dict, Tuple, Any

random.seed(333)

def state_dict_prune(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, (int, float)):
            new_state_dict[key] = str(type(value))
        else:
            new_state_dict[key] = value
    
    return new_state_dict

def simple_math_program_exec(program: str, return_dict) -> Any:
    l = {}
    try:
        exec(program, {}, l)
        return_dict['vars'] = l
    except:
        pass
def get_gsm_output_state_unsafe(program: str) -> Dict[str, Any]:
    if "for " in program or "while " in program:
        return get_gsm_output_state(program)

    return_dict = {}
    simple_math_program_exec(program, return_dict)

    if len(return_dict) == 0 or "vars" not in return_dict:
        return None
    else:
        return return_dict["vars"]

def get_gsm_output_state(program: str) -> Dict[str, Any]:
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=simple_math_program_exec, args=(program, return_dict))
    p.start()
    p.join(timeout=5)

    if len(return_dict) == 0 or "vars" not in return_dict:
        return None
    else:
        return return_dict["vars"]

def get_gsm_answer(metadata: Dict[str, Any]) -> float:
    answer_str = metadata['answer'].split('#### ')[1].replace(",", "")
    return float(answer_str)

def process_gsm_output_example(code: str, metadata: dict, 
                               distinct_programs_counts: dict = None) -> dict:
    # process the raw code outputs
    code = "\n".join(list(filter(lambda x: len(x.strip()) > 0, code.split("\n"))))
    code = code.split("\n\n")[0].strip()
    code = code.replace("\"", "'")
    lower_code = re.sub(r"\b(?<!')(\w+)(?!')\b", lambda match: match.group(1).lower(), code) 

    # do not emit the same program
    if distinct_programs_counts is not None:
        if lower_code in distinct_programs_counts:
            distinct_programs_counts[lower_code] += 1
            return None
        else:
            distinct_programs_counts[lower_code] = 1
    
    exec_result = get_gsm_output_state_unsafe(code)
    if exec_result is None:
        execution_match = False
    else:
        exec_result = state_dict_prune(exec_result)
        if "answer" not in exec_result:
            execution_match = False
        else:
            gold_answer = get_gsm_answer(metadata)
            try:
                execution_match = float(exec_result["answer"]) == gold_answer
            except Exception as e:
                execution_match = False
    
    # construct the program dictionary
    program_dict = {"code": code,
                    "lower_code": lower_code,
                    "exec_result": exec_result, 
                    "exec_match": execution_match}
    
    return program_dict

def postprocess_gsm_output(input_file_paths: List[str], output_file_path: str):
    examples = []
    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as f:
            examples.extend([json.loads(s) for s in f.readlines()])

    f = open(output_file_path, "w+")
    
    for output in tqdm(examples):
        metadata = output["example"]
        result_example_dict = {"metadata": metadata}

        processed_programs = []
        distinct_programs_counts = {}

        # add all the generated results
        # NOTE: the generated programs may be the same as the gold one, but we keep them separated during processing
        for code in output["generated_output"]: 
            # process the raw code outputs
            processed_result = process_gsm_output_example(code, metadata, distinct_programs_counts)

            if processed_result is not None:
                processed_programs.append(processed_result)
            else:
                continue
        
        # add the program count
        for processed_program in processed_programs:
            processed_program["program_count"] = distinct_programs_counts[processed_program["lower_code"]]
        
        result_example_dict["generated_programs"] = processed_programs

        # write to file
        f.write(json.dumps(result_example_dict) + "\n")
    
    f.close()

def split_train_dev():
    examples = []
    with open("data/gsmath/codex_40_output_train_dev.jsonl", "r") as f:
        examples.extend([json.loads(s) for s in f.readlines()])
    print(f"{len(examples)} examples")

    random.shuffle(examples)

    train_examples = examples[:int(len(examples) * 0.8)]
    dev_examples = examples[int(len(examples) * 0.8):]

    with open("data/gsmath/codex_40_output_train.jsonl", "w+") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")
        print(f"{len(train_examples)} train examples saved")

    with open("data/gsmath/codex_40_output_dev.jsonl", "w+") as f:
        for example in dev_examples:
            f.write(json.dumps(example) + "\n")
        print(f"{len(dev_examples)} dev examples saved")

def split_train_dev_from_codex_output():
    examples = []
    with open("data/gsmath/train.jsonl", "r") as f:
        examples.extend([json.loads(s) for s in f.readlines()])
    print(f"{len(examples)} examples")

    codex_train_examples = []
    with open("data/gsmath/codex_40_output_train.jsonl", "r") as f:
        codex_train_examples.extend([json.loads(s) for s in f.readlines()])

    codex_dev_examples = []
    with open("data/gsmath/codex_40_output_dev.jsonl", "r") as f:
        codex_dev_examples.extend([json.loads(s) for s in f.readlines()])

    # assert len(codex_train_examples) + len(codex_dev_examples) == len(examples)

    with open("data/gsmath/split_train.jsonl", "w") as f:
        for example in codex_train_examples:
            f.write(json.dumps(example["metadata"]) + "\n")

    with open("data/gsmath/split_dev.jsonl", "w") as f:
        for example in codex_dev_examples:
            f.write(json.dumps(example["metadata"]) + "\n")

if __name__ == "__main__":
    # train_files = [f"results_cot/gsm_codex_davinci-few_shot_baseline-for-in_house_models-pass@40-gsm_shot-train-{i}.jsonl" for i in range(1, 3)]
    # test_files = ["results_cot/gsm_codex_davinci-few_shot_baseline-for-in_house_models-pass@40-gsm_shot-test.jsonl"]
    # postprocess_gsm_output(train_files, "data/gsmath/codex_40_output_train.jsonl")

    split_train_dev_from_codex_output()