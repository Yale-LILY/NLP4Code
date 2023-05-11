import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="The dataset to evaluate on.")
args = parser.parse_args()

dataset = args.dataset

print("Starting report on dataset: {}".format(dataset))

failed_options = ['missing', 'extra', 'subtle', 'unclear']
successfull_options = ['spurious', 'same', 'different']
error_options = ['ERROR: program failed to execute', 'ERROR: no answer variable']

failed_dict = {}
success_dict = {}
error_dict = {}

for key in failed_options:
    failed_dict[key] = []

for key in successfull_options:
    success_dict[key] = []

for key in error_options:
    error_dict[key] = []

with open(dataset, 'r') as f:
    examples = [json.loads(s) for s in f.readlines()]

    for example in examples[1:]:       # the first example is the header
        evaluation = example['evaluation']
        if evaluation in failed_options:
            failed_dict[evaluation].append(example)

        elif evaluation in successfull_options:
            success_dict[evaluation].append(example)

        elif evaluation in error_options:
            error_dict[evaluation].append(example)


print("\033[1;31mNumber of failed examples: \033[0m")
failed = 0
for key in failed_dict:
    failed += len(failed_dict[key])
    print("\t{}: {}".format(key, len(failed_dict[key])))
print("Total Failed: {}".format(failed))

print("\033[1;32mNumber of successfull examples: \033[0m")
success = 0
for key in success_dict:
    success += len(success_dict[key])
    print("\t{}: {}".format(key, len(success_dict[key])))
print("Total Success: {}".format(success))


print("\033[1;33mNumber of error examples: \033[0m")
error = 0
for key in error_dict:
    error += len(error_dict[key])
    # currently all errors are being saved under one header
print("Total Errors: {}".format(error))
