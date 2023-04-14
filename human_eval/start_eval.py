import argparse

from human_eval.evaluation_tasks import EvaluationTask

def main():
    """calling start_eval.py takes two input variables, dataset and output_file.
        dataset: this dataset contains a model's outputs. given in the form of a path such as \data\squall\squall_processed_dev_all.jsonl
        output_file: this lets the user specify where to output the human evals to. This is also a path"""


    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="The dataset to evaluate on.")
    parser.add_argument("output_file", type=str, help="The output file to save the evaluation results.")
    args = parser.parse_args()

    dataset = args.dataset
    output_file = args.output_file

    print("Starting evaluation on dataset: {}".format(dataset))
    print("Saving results to: {}".format(output_file))

    # we initialize the task as an EvaluationTask, defined in evaluation_tasks.py
    # this class has several functions that help with the evaluation
    task = EvaluationTask(dataset_path=dataset, output_file=output_file)

    same_example = False        # Used to keep track if a problem's been seen before. 
    last_evaluation = None
    while True:
        if not same_example:
            example = task.get_and_display_next_example()
            # print(example['metadata'])

        answer = example['generated_program'].get('exec_match', example['generated_program'].get('exec_acc', False))

        if answer == "ERROR: no answer variable":        # sometimes there is an error in the generated program. This will be parsed automatically later.
            print("\033[1;33m ERROR on answer.")
            task.save_single_evaluation(example['metadata'], "ERROR")       # we simply save it as an error for now.
            same_example = False

        elif answer:
            print("\033[1;32m" +  " Reason for success: " + "\033[0m")
            print("\tSpurious: 0")
            print("\tSame as gold: 1")
            print("\tdifferent from gold: 2")
            reason = input("Enter the number corresponding to the reason for success: ")
            reasons = ['spurious', 'same', 'different']
            task.save_single_evaluation(example['metadata'], reasons[int(reason)])
            same_example = False

        else:
            print("\033[1;31m" +  " Reason for failure: " + "\033[0m")
            print("\tMissing something: 0")
            print("\tSomething extra: 1")
            print("\tSubtle difference: 2")
            reason = input("Enter the number corresponding to the reason for failure: ")
            reasons = ['missing', 'extra', 'subtle']
            task.save_single_evaluation(example['metadata'], reasons[int(reason)])
            same_example = False


if __name__ == '__main__':
    main()