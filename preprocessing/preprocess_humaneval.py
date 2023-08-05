import json
import os
import random
from human_eval.data import write_jsonl, read_problems

file_path = os.getcwd() + "/data/humaneval/humaneval.jsonl"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

problems = read_problems()
result_jsonl = []
metadata_keys = ['task_id', 'entry_point', 'test']
for task_id in problems:
    new_dict = {}
    metadata = {}
    for key in metadata_keys:
        metadata[key] = problems[task_id][key]
    new_dict['metadata'] = metadata
    new_dict['prompt'] = problems[task_id]['prompt']
    new_dict['canonical_solution'] = problems[task_id]['canonical_solution']
    result_jsonl.append(new_dict)

write_jsonl(file_path, result_jsonl)

selected_problems = []
with open(file_path, 'r') as file:
    for line in file:
        json_obj = json.loads(line)
        selected_problems.append(json_obj)
selected_problems = random.sample(selected_problems, 8)

file_path = os.getcwd() + "/prompt_files/humaneval-8_examplars.jsonl"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

write_jsonl(file_path, selected_problems)
