# Removes any characters in `chars_to_remove` from the front of `s`
from tracemalloc import start


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
    if sql_query[idx] == " ":
        print("[is_idx_at_token_start] idx not in word")
        return False

    if not sql_query[idx-1] == " ":
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
