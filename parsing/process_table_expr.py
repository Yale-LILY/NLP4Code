from clean_query import remove_consecutive_spaces
from helpers import get_next_token_idx, get_cur_token, get_prev_token, get_next_token
import re


# Given a SQL query with exactly one SELECT, extract table FROM which query is answered
def extract_table_expr_from_query(simple_sql_query: str):
    simple_sql_query = remove_consecutive_spaces(simple_sql_query)
    start_idx = simple_sql_query.find("FROM ")
    if start_idx < 0:
        print("[extract_table] no FROM in simple_sql_query")
        return None

    start_idx += len("FROM ")
    while start_idx < len(simple_sql_query) and simple_sql_query[start_idx] == " ":
        start_idx += 1
    idx = get_next_token_idx(simple_sql_query, start_idx)
    while idx < len(simple_sql_query):
        cur_word = get_cur_token(simple_sql_query, idx)
        if cur_word == "JOIN":
            idx = get_next_token_idx(simple_sql_query, idx)
            idx = get_next_token_idx(simple_sql_query, idx)
        elif cur_word == "AS":
            idx = get_next_token_idx(simple_sql_query, idx)
            idx = get_next_token_idx(simple_sql_query, idx)
        elif cur_word == "ON":
            idx = get_next_token_idx(simple_sql_query, idx)
            idx = simple_sql_query.find("=", idx)
            if idx < 0:
                return None
            idx += 1
            while simple_sql_query[idx] == " ":
                idx += 1
            idx = get_next_token_idx(simple_sql_query, idx)
        else:
            return simple_sql_query[start_idx:idx].strip()

    return simple_sql_query[start_idx:idx].strip()


def extract_table_aliases(sql_table_expr: str):
    table_alias_dict = dict()

    idx = 0
    while idx < len(sql_table_expr):
        cur_word = get_cur_token(sql_table_expr, idx)
        if cur_word == "AS":
            table_name = get_prev_token(sql_table_expr, idx)
            alias_name = get_next_token(sql_table_expr, idx)
            table_alias_dict.setdefault(alias_name, table_name)
            idx = get_next_token_idx(sql_table_expr, idx)
            idx = get_next_token_idx(sql_table_expr, idx)
        else:
            idx = get_next_token_idx(sql_table_expr, idx)

    return table_alias_dict


def substitute_symbol_for_table_expr(simple_sql_query: str, sql_table_expr: str, sub_symbol: str):
    idx = simple_sql_query.find(sql_table_expr)
    if idx < 0:
        print("[substitute_symbol_for_table] sql_table_expr not in simple_sql_query")
        return simple_sql_query

    return re.sub(sql_table_expr, sub_symbol, simple_sql_query)


def sql_table_expr_to_pandas(sql_table_expr: str) -> str:
    return ""
