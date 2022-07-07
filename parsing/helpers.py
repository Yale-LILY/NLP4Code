from typing import Tuple


# Removes any characters in `chars_to_remove` from the front of `s`
def trim_front(s, chars_to_remove):
    while s[0] in chars_to_remove:
        s = s[1:]
    return s


# Removes any characters in `chars_to_remove` from the back of `s`
def trim_back(s, chars_to_remove):
    while s[-1:] in chars_to_remove:
        s = s[:-1]
    return s


# Removes characters like parentheses from front/end of `s`
def trim_front_and_back(s, char_front, char_back):
    while s[0] == char_front and s[-1:] == char_back:
        s = s[1:-1]
    return s


# Find corresponding balanced closing parenthesis for opening parenthesis at index `open_idx-1`
def find_closing_parenthesis(s, open_idx):
    if s[open_idx-1] != '(':
        print('[find_closing_parenthesis] input open_idx error')
        return -1

    idx = open_idx
    ct_open = 0
    while idx < len(s):
        if s[idx] == '(':
            ct_open += 1
        elif s[idx] == ')':
            if ct_open == 0:
                return idx
            ct_open -= 1

        idx += 1

    return -1


# Determines if first non-whitespace char in `partial_sql_query` is "SELECT"
def is_next_token_select(partial_sql_query):
    return partial_sql_query.strip().find("SELECT") == 0


def is_idx_at_token_start(sql_query: str, idx: int):
    if idx >= len(sql_query):
        return False

    if sql_query[idx] == " ":
        print("[is_idx_at_token_start] idx not in word")
        return False

    if idx > 0 and not sql_query[idx-1] == " ":
        print("[is_idx_at_token_start] idx not at start of token")
        return False

    return True


def get_next_token_idx(sql_query: str, idx: int):
    while idx < len(sql_query) and sql_query[idx] != " ":
        idx += 1

    while idx < len(sql_query) and sql_query[idx] == " ":
        idx += 1

    return idx


def get_prev_token(sql_query: str, idx: int):
    if idx == 0:
        print("[get_prev_token] no prev token")
        return None

    if not is_idx_at_token_start(sql_query, idx):
        return None

    finish_idx = idx - 1
    while finish_idx - 1 >= 0 and sql_query[finish_idx-1] == " ":
        finish_idx -= 1

    start_idx = finish_idx - 1
    while start_idx - 1 >= 0 and sql_query[start_idx - 1] != " ":
        start_idx -= 1

    return sql_query[start_idx:finish_idx]

def get_second_last_token(sql_query: str):
    length = len(sql_query)
    if length < 2:
        print("[get_second_last_token] no second last token")
        return None

    finish_idx = length - 1
    while finish_idx > 0 and sql_query[finish_idx] != " ":
        finish_idx -= 1

    start_idx = finish_idx - 1
    while start_idx > 0 and sql_query[start_idx] != " ":
        start_idx -= 1

    return sql_query[start_idx:finish_idx].strip()

def get_cur_token(sql_query: str, idx: int):
    if not is_idx_at_token_start(sql_query, idx):
        return None

    finish_idx = idx
    while finish_idx < len(sql_query) and sql_query[finish_idx] != " ":
        finish_idx += 1

    return sql_query[idx:finish_idx]


def get_next_token(sql_query: str, idx: int):
    if idx >= len(sql_query) - 1:
        print("[get_prev_token] no next token")
        return None

    if not is_idx_at_token_start(sql_query, idx):
        return None

    start_idx = get_next_token_idx(sql_query, idx)
    return get_cur_token(sql_query, start_idx)


def remove_prev_token(s: str, idx: int) -> Tuple[str, int]:
    """Removes previous token from idx, where idx is at start of token.

    Args:
        s (str): String from which to remove previous token
        idx (int): Index of start of token, where previous token from idx is removed.

    Returns:
        Tuple[str, int]: Redacted string, and new position of idx
    """
    if idx == 0:
        print("[get_prev_token] no prev token")
        return None

    if not is_idx_at_token_start(s, idx):
        return None

    finish_idx = idx - 1
    while finish_idx - 1 >= 0 and s[finish_idx-1] == " ":
        finish_idx -= 1

    start_idx = finish_idx - 1
    while start_idx - 1 >= 0 and s[start_idx - 1] != " ":
        start_idx -= 1

    return s[:start_idx] + s[finish_idx:], idx - (finish_idx - start_idx)


def extract_table_column(join_on_col: str) -> str:
    """For a table column of the form TABLE.COLUMN (as in JOIN), extract COLUMN.

    Args:
        join_on_col (str): Full name of column, potentially with table specified.

    Returns:
        str: Extracted column (without specified table, if specified).
    """
    dot_idx = join_on_col.find(".")
    return join_on_col if dot_idx < 0 else join_on_col[dot_idx+1:]


def get_first_token(s: str) -> str:
    idx = s.find(" ")
    if idx < 0:
        idx = len(s)
    return s[:idx]

def subtract_sql_to_pandas(sql: str, simple: bool) -> str:
    """If simple subtract, removes the SELECT and parenthesis and ; from the sql for a subtract sql 
    Otherwise, replaces subtraction with pandas and returns the new sql with subtraction replaced

    TODO fill args
    """
    if simple:
        ret = sql.replace("SELECT ", "").replace(";", "").replace("(", "").replace(")", "")
    else:
        ret = None
    return ret