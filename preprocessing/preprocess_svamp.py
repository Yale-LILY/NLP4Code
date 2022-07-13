import json

from typing import List, Dict, Any
from csv import DictReader


input_file = "./data/svamp/svamp.json"
dev_file = "./data/svamp/svamp.jsonl"
# keywords = ["ID", "Body", "Question", "Equation", "Answer"]

def get_code_from_answer_str(answer_str: str, question_str: str) -> str:
    pass

def process_svamp(instances: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    for i, instance in enumerate(instances):
        # put it in the mathqa style: text, code, answer, task_id
        body = instance["Body"]
        question = instance["Question"]
        instance["text"] = f"{body} {question}"
        instance.pop("Question")
        instance.pop("Body")

        instance["task_id"] = f"chal_{i}"
        instance.pop("ID")

        instance["original_answer"] = instance["Answer"]
        instance.pop("Answer")

        instance["code"] = get_code_from_answer_str(instance["Equation"], instance["original_answer"])

        instance.pop("Type")

        print(instance)

if __name__ == "__main__":
    # load the dev data
    with open(input_file) as in_file:
        lines = json.load(in_file)
 
    processed_data = process_svamp(lines)

    # write processed data to file
    with open("./annotated data/svamp/svamp.jsonl", "w") as f:
        f.write("\n".join([json.dumps(data) for data in processed_data]))
        # f.write(str(dev_lines))