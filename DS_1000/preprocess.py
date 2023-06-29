from ds1000 import DS1000Dataset
import json
import random

ds_data = DS1000Dataset("ds1000_data", mode="Completion") # loads all questions into RAM

keys = ["reference_code", "prompt"]
libraries = ["Matplotlib", "Numpy", "Pandas", "Pytorch", "Scipy", "Sklearn", "Tensorflow"]

metadata_keys = ['lib', 'perturbation_type', 'perturbation_origin_id', 'source_url']

for lib in libraries:
    for id, dictionary in enumerate(ds_data[lib]):
        new_dict = {}
        metadata = {}
        for key in metadata_keys:
            metadata[key] = dictionary[key]
        new_dict["metadata"] = metadata

        for key in keys:
            new_dict[key] = dictionary[key]

        with open("ds1000.jsonl", 'a') as file:
            json.dump(new_dict, file)
            file.write('\n')

ds1000_jsonl = []
with open("ds1000.jsonl", 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            ds1000_jsonl.append(json_obj)
selected_problems = random.sample(ds1000_jsonl, 8)
for problem in selected_problems:
     with open("../prompt_files/ds1000-8_exemplars.jsonl", 'a') as file:
            json.dump(problem, file)
            file.write('\n')

