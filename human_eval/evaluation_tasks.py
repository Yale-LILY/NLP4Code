import json
import os
import random
import time

from typing import List, Dict, Any

class EvaluationTask:
    def __init__(self, dataset_path: str, output_file: str, evaluation_size: int=100):
        self.dataset_path = dataset_path
        self.examples = []
        self.read_data()
        self.output_file = output_file
        # self.postprocess_data()

        # create the output file if it does not exist
        if not os.path.isfile(output_file):
            print("Creating new output file")
            self.evaluation_indices = self.get_evaluation_indices()
            print(f"Creating new output file {output_file} for {len(self.evaluation_indices)} examples")
            with open(output_file, "w+") as f:
                evaluation_metadata = {"data_file": self.dataset_path, 
                                       "total_num_examples": len(self.examples),
                                       "evaluation_indices": self.evaluation_indices}
                f.write(json.dumps(evaluation_metadata) + "\n")

            # keep track of the progress
            self.evaluated_examples = []
        else:
            # recovery the metadata and the evaluated examples
            print("Recovering progress from existing output file")
            self.recovery_progress(output_file)

    def read_data(self):
        with open(self.dataset_path, "r") as f:
            self.examples = [json.loads(s) for s in f.readlines()]

    def get_evaluation_indices(self) -> List[int]:
        indicies = list(range(len(self.examples)))
        random.shuffle(indicies)            # shuffle the indicies to get a random sample
        return indicies

    def recovery_progress(self, output_file: str):
        # first recover the evaluation progress from the file
        with open(output_file, "r") as f:
            lines = f.readlines()
            evaluation_metadata = json.loads(lines[0])
            self.evaluated_examples = [json.loads(s) for s in lines[1:]]
            self.evaluation_indices = evaluation_metadata["evaluation_indices"]
        
        # then verify the evaluated examples to match the data file
        for i, example in enumerate(self.evaluated_examples):
            assert example["metadata"] == self.examples[self.evaluation_indices[i]], \
                f"evaluated example does not match the data file"
        
        print(f"Recovered progress from {output_file} for {len(self.evaluated_examples)} out of {len(self.evaluation_indices)} total examples")
        time.sleep(2)

    def save_single_evaluation(self, example: Dict[str, Any], evaluation: str):
        save_example = {"metadata": example, "evaluation": evaluation}

        # save both to the output file and the evaluated examples
        self.evaluated_examples.append(save_example)
        with open(self.output_file, "a") as f:
            f.write(json.dumps(save_example) + "\n")

    def get_and_display_next_example(self):
        next_example_idx = self.evaluation_indices[len(self.evaluated_examples)]
        print("\033[1;7;34m" + '#' * 20 + f" Example {next_example_idx} " + '#' * 20 + "\033[0m")
        self.display_example(self.examples[next_example_idx])
        # print("\033[1;7;34m" + '#' * 40 + "\033[0m")
        print('#' * 40)
        return self.examples[next_example_idx]

    def display_example(self, example: Dict[str, Any]) -> str:
        # prints the metadata. We're currently assuming that the dataset is spider or squall
        if example['generated_program']['exec_acc']:
            print("\033[1;32m" + "Execution Accuracy: True" + "\033[0m")
        else:
            print("\033[1;31m" + "Execution Accuracy: False" + "\033[0m")

        print(f"Question: {example['metadata']['question']}")
        print(f"Generated Program: {example['generated_program']['program']}")
        print("Tables:")
        for header in example['metadata']['db_table_headers']:
            print("Table: ", header)
            for row in example['metadata']['db_table_headers'][header]:
                print("\t" + row)