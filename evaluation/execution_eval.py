import math
from utils.parsing_utils import get_statements_from_code

def parseable(code):
    stmts = get_statements_from_code(code)
    if stmts is not None:
        return True
    else:
        return False

def stmt_count(code: str):
    stmts = get_statements_from_code(code)
    if stmts is not None:
        return len(stmts)
    else:
        return 0

def perplexity(loss: float):
    """ When NLLLoss is used, the token-level perplexity is simply the exp of token-level NLLLoss """
    return math.exp(float(loss))