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
        elif annotation == "override":
            exec_match, exec_info = task.check_annotation_correctness(example, last_annotation)
            task.save_single_annotation(example, last_annotation, exec_result=exec_info)
            same_example = False
        else: 
            exec_match, exec_info = task.check_annotation_correctness(example, annotation)
            if exec_match:
                task.save_single_annotation(example, annotation, exec_result=exec_info)
                same_example = False
                save = input("Correct! Press ENTER to save this annotation, or type `cancel` to improve this annotation: ")
                if save == "cancel":
                    same_example = True
                    last_annotation = annotation
            else:
                last_annotation = annotation
                print("Annotation is not correct. More information: (if you believe the annotation is correct, enter `override`)")
                print(exec_info)
                same_example = True
            

if __name__ == '__main__':
    main()