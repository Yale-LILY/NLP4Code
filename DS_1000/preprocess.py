from ds1000 import DS1000Dataset
import json

ds_data = DS1000Dataset("ds1000_data", mode="Completion") # loads all questions into RAM

keys = ["reference_code", "prompt"]
libraries = ["Matplotlib", "Numpy", "Pandas", "Pytorch", "Scipy", "Sklearn", "Tensorflow"]

metadata_keys = ['lib', 'perturbation_type', 'perturbation_origin_id', 'source_url']

print(ds_data["Numpy"][0]["prompt"])
is_correct = ds_data["Numpy"][0].test("result = a.shape")
print(is_correct)
exit()

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