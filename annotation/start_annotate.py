import os
import time

from annotation.annotation_tasks import SQL2PandasAnnotationTask

ANNOTAION_TASKS = ["spider", "squall"]

def main():
    input("Press Enter to start annotation...")
    task_name = input("Select the dataset to annotate (spider/squall): ")
    task = SQL2PandasAnnotationTask(dataset_name=task_name)

    same_example = False
    last_annotation = None
    while True:
        if not same_example:
            example = task.get_and_display_next_example()
        annotation = input(task.get_annotation_instructions(example))
        if annotation == "exit":
            break
        elif annotation == "skip":
            task.save_single_annotation(example, annotation)
            same_example = False
        elif annotation == "override" and last_annotation is not None:
            exec_match, exec_info = task.check_annotation_correctness(example, last_annotation)
            task.save_single_annotation(example, last_annotation, exec_result=exec_info)
            same_example = False
        else: 
            exec_match, exec_info = task.check_annotation_correctness(example, annotation)
            last_annotation = annotation
            if exec_match:
                same_example = False
                print("\033[1m" + " RESULT... " + "\033[0m")
                print("\033[42m" + "Correct! Returned: " + exec_info + "\033[0m")
                save = input("Press \033[33mENTER\033[0m to save this annotation, or type `\033[33mcancel\033[0m` to improve this annotation: ")
                print("_________________________________________________________________________________________________________")
                if save == "cancel":
                    same_example = True
                else:
                    task.save_single_annotation(example, annotation, exec_result=exec_info)
            else:
                print("\033[1m" + " RESULT... " + "\033[0m")
                print("\033[41m" + exec_info + "\033[0m")
                print("Annotation is not correct. More information: (if you believe the annotation is correct, enter `\033[33moverride\033[0m`)")
                print("_________________________________________________________________________________________________________")
                same_example = True
            

if __name__ == '__main__':
    main()