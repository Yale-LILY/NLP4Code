import argparse
import json


def main():
    """calling start_eval.py takes two input variables, dataset and output_file.
        dataset: this dataset contains a model's outputs. given in the form of a path such as \data\squall\squall_processed_dev_all.jsonl
        output_file: this lets the user specify where to output the human evals to. This is also a path"""


    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=str, help="The first json file for the dataset evaluated.")
    parser.add_argument("source_file", type=str, help="The json file for the source dataset.")
    args = parser.parse_args()

    original = args.original
    source_file = args.source_file

    with open(original, 'r') as f:
        original = [json.loads(s) for s in f.readlines()]

    with open(source_file, 'r') as f:
        new = [json.loads(s) for s in f.readlines()]

    indices = []

    for example in original[1:100]:        # all items in the array but the first one
        print(example)
        for i, new_example in enumerate(new):
            if example['metadata']['question'] == new_example['metadata']['question']:
                indices.append(i)
    
    print(indices)

if __name__ == "__main__":
    main()