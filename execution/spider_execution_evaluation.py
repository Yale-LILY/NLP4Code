import multiprocessing.pool
import ast

from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor as Pool
from execution.safe_execution_util import execute

DB_DIR = "data/spider/database"

def batch_exec_programs(programs: List[str], exec_func: callable, n_processes: int = 20, db_ids: str = None) -> List[Any]:
    # build a dict to optimize for potential same programs

    with Pool(n_processes) as p:
        executed_answers = p.starmap(exec_func, zip(programs, db_ids))
    executed_answers = list(executed_answers)
    return executed_answers, len(programs)

def batch_execution_acc(programs: List[str], exec_func: callable, answers: List[str], 
                        n_examples: int, eval_at_k: int, answer_eq_func: callable, db_ids: str,
                        n_processes: int = 20) -> List[Tuple[float, float]]:
    """
    This function evaluates execution accuracy for a batch of programs using multiprocessing.

    Returns: execution accuracy, execution rate
    """
    assert len(programs) == len(answers) * eval_at_k
    assert n_examples * eval_at_k == len(programs)
    
    executed_answers, n_programs = batch_exec_programs(programs, exec_func, n_processes, db_ids)

    print(f"Evaluating {len(programs)} generated programs for {n_examples} tasks")

    # separate the results for each task
    grouped_executed_answers = [executed_answers[i*eval_at_k:(i+1)*eval_at_k] for i in range(0, n_examples)]
    grouped_execution_evals = []
    for predicted_answers, gold_answer in zip(answers, executed_golden_answers):
        correct_count = 0.0
        for predicted_answer in predicted_answers:
            if answer_eq_func(predicted_answer, gold_answer):
                correct_count += 1

        accuracy_at_k = correct_count / eval_at_k
        pass_at_k = correct_count > 0.0

        grouped_execution_evals.append((accuracy_at_k, pass_at_k))

    return grouped_execution_evals

def execution_acc(program: str, exec_func: callable, answer: str, answer_eq_func: callable, db_id: str = "") -> Tuple[float, float]:
    """
    This function is used to evaluate the accuracy of the execution of the program.

    Returns: execution accuracy, execution rate
    """

    executed_answer = exec_func(program, db_id)

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
 