from ds1000 import DS1000Dataset
import json
import random
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

ds_data = DS1000Dataset("ds1000_data", mode="Completion") # loads all questions into RAM

keys = ["reference_code", "prompt"]
libraries = ["Matplotlib", "Numpy", "Pandas", "Pytorch", "Scipy", "Sklearn", "Tensorflow"]

metadata_keys = ['lib', 'perturbation_type', 'perturbation_origin_id', 'source_url']

with open("../data/ds1000/ds1000.jsonl", "w") as file:
    file.write("")

for lib in libraries:
    for id, dictionary in enumerate(ds_data[lib]):
        # if len(tokenizer.tokenize(dictionary['prompt'])) > 2048:
        #     continue
        new_dict = {}
        metadata = {}
        for key in metadata_keys:
            metadata[key] = dictionary[key]
        metadata['id'] = id
        new_dict["metadata"] = metadata

        for key in keys:
            new_dict[key] = dictionary[key]

        with open("../data/ds1000/ds1000.jsonl", 'a') as file:
            json.dump(new_dict, file)
            file.write('\n')


with open("../prompt_files/ds1000-7_exemplars.jsonl", "w") as file:
    file.write("")
selected_problems = []
last_lib = ""
with open("../data/ds1000/ds1000.jsonl", 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            if json_obj["metadata"]["lib"] != last_lib:
                selected_problems.append(json_obj)
                last_lib = json_obj["metadata"]["lib"]

for problem in selected_problems:
     with open("../prompt_files/ds1000-7_exemplars.jsonl", 'a') as file:
            json.dump(problem, file)
            file.write('\n')

