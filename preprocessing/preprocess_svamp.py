"""Preprocessing script for SVAMP. 

A typical example of SVAMP look like this:
{
    "ID": "chal-1",
    "Body": "Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack",
    "Question": "How much do you have to pay to buy each pack?",
    "Equation": "( 76.0 - 25.0 )",
    "Answer": 51.0,
    "Type": "Subtraction"
},

And after preprocessing, we want it to look like this:
{
    "question": "Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack. How much do you have to pay to buy each pack?",
    "answer": 51.0,
    "annotated_code": <only available for prompt examples>,
    "metadata": {
                    "ID": "chal-1",
                    "Body": "Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack",
                    "Question": "How much do you have to pay to buy each pack?",
                    "Equation": "( 76.0 - 25.0 )",
                    "Answer": 51.0,
                    "Type": "Subtraction"
                },
}

"""

import json

from typing import Dict, List, Any


ANNOTATION_DICT = {
    "chal-10": "n_customers_left = 9\nn_customers_now = 12\nn_customers_start = n_customers_now + n_customers_left\nanswer = n_customers_start",
    "chal-11": "n_birds = 3\nn_storks = 6\nn_more_bird = 2\nn_more_stork_than_bird = n_storks - (n_birds + n_more_bird)\nanswer = n_more_stork_than_bird",
    "chal-12": "n_tables = 11\nn_chairs_per_table = 13\nn_chairs = n_tables * n_chairs_per_table\nanswer = n_chairs",
    "chal-23": "group_size = 18\nn_total_bananas = 180\nn_groups = n_total_bananas / group_size\nanswer = n_groups",
}

def preprocess_svamp_instance(example: Dict[str, Any]) -> Dict[str, Any]:
    # preprocess based on the example
    preprocessed_example = {}
    preprocessed_example["question"] = example["Body"] + (" " if example["Body"].endswith(".") else ". ") \
        + example["Question"]
    preprocessed_example["answer"] = example["Answer"]
    preprocessed_example["metadata"] = example

    return preprocessed_example

def main():
    with open("data/svamp/SVAMP.json", "r") as f:
        examples = json.load(f)
    
    print(f"loaded {len(examples)} examples")

    # preprocess the examples
    processed_examples = [preprocess_svamp_instance(example) for example in examples]

    # split the examples to prompt and test sets
    prompt_examples = list(filter(lambda x: x["metadata"]["ID"] in ANNOTATION_DICT, processed_examples))
    test_examples = list(filter(lambda x: x["metadata"]["ID"] not in ANNOTATION_DICT, processed_examples))

    # save the program annotations to the prompt examples
    for example in prompt_examples:
        example["annotated_code"] = ANNOTATION_DICT[example["metadata"]["ID"]]

    # save the prompt and test sets
    print(f"Saving {len(prompt_examples)} prompt examples and {len(test_examples)} test examples")
    with open("svamp-idiomatic_code-annotated-4_exemplars.jsonl", "w+") as f:
        for example in prompt_examples:
            f.write(json.dumps(example) + "\n")
    
    with open("data/svamp/svamp_test.jsonl", "w+") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    main()