import json
import random
from ds1000 import DS1000Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

ds_data = DS1000Dataset("ds1000_data", mode="Completion") # loads all questions into RAM

keys = ["reference_code", "prompt"]
libraries = ["Matplotlib", "Numpy", "Pandas", "Pytorch", "Scipy", "Sklearn", "Tensorflow"]

metadata_keys = ['lib', 'perturbation_type', 'perturbation_origin_id', 'source_url']

with open("../data/ds1000/ds1000.jsonl", 'w') as file:
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

            json.dump(new_dict, file)
            file.write('\n')

selected_problems = []
with open("../data/ds1000/ds1000.jsonl", 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            selected_problems.append(json_obj)

selected_problems = random.sample(selected_problems, 4)

with open("../prompt_files/ds1000-4_exemplars.jsonl", "w") as file:
    for new_dict in selected_problems:
        json.dump(new_dict, file)
        file.write('\n')