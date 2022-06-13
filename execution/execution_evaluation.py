import ast
import inspect

from typing import List, Dict, Tuple, Any
# from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import Pool
from execution.safe_execution_util import execute
from itertools import chain

def batch_exec_programs(programs: List[str], exec_func: callable, n_processes: int = 20, 
                        extra_exec_args: Dict[str, List[Any]]={}, dedup: bool = False) -> List[Any]:
    assert dedup == False or len(extra_exec_args) == 0, "dedup is not supported for extra_exec_args"
    if dedup:
        # build a dict to optimize for potential same programs
        program_dict = {}
        for program in programs:
            if program not in program_dict:
                program_dict[program] = None
        unique_programs = list(program_dict.keys())

        idx = 0
        parsable_unique_programs = []
        for program in unique_programs:
            try:
                ast.parse(program, mode="exec")
                parsable_unique_programs.append(program)
                program_dict[program] = idx
                idx += 1
            except SyntaxError:
                program_dict[program] = -1
            except MemoryError:
                print(f"MemoryError when parsing {program}")
                program_dict[program] = -1
    else:
        parsable_unique_programs = programs

    # make sure that the extra_exec_args matches the function signature
    extra_exec_args_list = []
    for arg_name, param_obj in list(inspect.signature(exec_func).parameters.items())[1:]: # first by default is the program
        if arg_name not in extra_exec_args:
            assert param_obj.default is not param_obj.empty, f"{arg_name} is missing in extra_exec_args"
            break
        else:
            extra_exec_args_list.append(extra_exec_args[arg_name])

    with Pool(n_processes) as p:
        func_args = list(zip(*[parsable_unique_programs, *extra_exec_args_list]))
        unique_executed_answers = p.starmap(exec_func, func_args)

    unique_executed_answers = list(unique_executed_answers)

    if dedup:
        # build the original programs answer list
        unique_executed_answers.append(None) # all syntax error will be assigned to None
        executed_answers = [unique_executed_answers[program_dict[program]] for program in programs]
    else:
        executed_answers = unique_executed_answers

    return executed_answers, len(set(programs))


def batch_eval_at_k(programs: List[List[str]], exec_func: callable, answers: List[str], eval_at_k: int, 
                    answer_eq_func: callable, extra_exec_args: Dict[str, List[Any]]={}, n_processes: int = 20,
                    ) -> List[List[bool]]:
    # make sure all the numbers match
    assert len(programs) == len(answers)
    assert all([len(v) == len(answers) for v in extra_exec_args.values()])

    # flatten everything
    flatten_programs = list(chain(*programs))
    n_examples = len(flatten_programs)

    # since the extra_exec_args is per task/prompt (i.e., as answers) and not per completion, we need to clone them k times
    for k in extra_exec_args.keys():
        extra_exec_args[k] = list(chain(*[[item] * eval_at_k for item in extra_exec_args[k]]))

    executed_answers, n_unique_programs = batch_exec_programs(flatten_programs, exec_func, n_processes, extra_exec_args=extra_exec_args)
    print(f"Evaluating {len(flatten_programs)} generated programs for {n_examples} tasks, " + \
           f"but only {n_unique_programs} unique programs")

    # separate the results for each task
    grouped_executed_answers = [executed_answers[i*eval_at_k:(i+1)*eval_at_k] for i in range(0, n_examples)]
    grouped_execution_evals: List[List[bool]] = []
    for predicted_answers, gold_answer in zip(grouped_executed_answers, answers):
        exec_match_list = []
        for predicted_answer in predicted_answers:
            if answer_eq_func(predicted_answer, gold_answer):
                exec_match_list.append(True)
            else:
                exec_match_list.append(False)
        grouped_execution_evals.append(exec_match_list)

    return grouped_execution_evals


def batch_execution_acc(programs: List[List[str]], exec_func: callable, answers: List[str], 
                        eval_at_k: int, answer_eq_func: callable, 
                        n_processes: int = 20, extra_exec_args: Dict[str, List[Any]]={}) -> List[Tuple[float, float]]:
    """
    This function evaluates execution accuracy for a batch of programs using multiprocessing.

    Returns: acc@k, pass@k
    """
    grouped_execution_evals: List[List[bool]] = batch_eval_at_k(programs, exec_func, answers, eval_at_k, 
                                                                answer_eq_func, extra_exec_args, n_processes)
    pass_acc_at_k_list = [(float(sum(exec_match_list)) / len(exec_match_list), float(sum(exec_match_list) > 0)) 
                        for exec_match_list in grouped_execution_evals]

    return pass_acc_at_k_list

def execution_acc(program: str, exec_func: callable, answer: str, answer_eq_func: callable) -> Tuple[float, float]:
    """
    This function is used to evaluate the accuracy of the execution of the program.

    Returns: execution accuracy, execution rate
    """
    executed_answer = exec_func(program)
    if executed_answer is not None and answer_eq_func(executed_answer, answer):
        return 1.0, 1.0
    elif executed_answer is not None:
        return 0.0, 1.0
    else:
        return 0.0, 0.0

def execution_eval_at_k(programs: List[str], exec_func: callable, answer_eq_func: callable, answer: str, k: int) -> Tuple[float, float]:
    """
    Assign 1.0 when at least one out of the k programs execute to the correct answer

    Returns: (accuracy_at_k, pass_at_k)
    """
    assert len(programs) >= k, "The number of programs should be larger than k"

    correct_count = 0.0
    with Pool(20) as p:
        executed_answers = p.map(exec_func, programs[:k])
    for executed_answer in executed_answers:
        if answer_eq_func(executed_answer, answer):
            correct_count += 1

    accuracy_at_k = correct_count / k
    pass_at_k = correct_count > 0.0

    return accuracy_at_k, pass_at_k
