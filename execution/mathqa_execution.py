import multiprocessing.pool
import ast

from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor as Pool
from execution.safe_execution_util import execute

def mathqa_answer_eq(prediction: Any, gold_answer: Any) -> bool:
    try:
        # if the execution result is a numpy array, valueError will be raised
        if prediction == gold_answer:
            return True
        else:
            return False
    except ValueError:
        return False

def mathqa_execution(program: str) -> Any:
    """
    for mathqa-python, we should be getting the answers from the "answer" variable in the local() variables
    """

    result = execute(program)
    
    if result["result"] == "passed":
        if "answer" in result["locals"]:
            executed_answer = result["locals"]["answer"]
        else:
            # FIXME: this is so ad-hoc
            executed_answer = -10000
    else:
        executed_answer = None

    return executed_answer