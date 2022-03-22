from typing import Dict
from clean_query import remove_consecutive_spaces
from helpers import get_next_token_idx, get_cur_token, get_prev_token, get_next_token
import re


# TODO:
# - handle different types of JOIN
# - remove original table names altogether from extracted table expression; i.e. only use AS aliases
# - actually convert table expressions (i.e. JOIN ONs) into pandas snippets


def extract_table_expr_from_query(simple_sql_query: str) -> str:
    """Extracts combined table expression (including JOIN/ON and AS table aliases) from SQL query.

    Args:
        simple_sql_query (str): Simple SELECT SQL query.

    Returns:
        str: Substring containing table expression of SQL query.
    """
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
        elif cur_word == "ON" or cur_word == "AND":
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


def extract_table_aliases(sql_table_expr: str) -> Dict[str, str]:
    """Extracts AS aliases for tables in table expression.

    Args:
        sql_table_expr (str): SQL table expression from which to create alias table.

    Returns:
        Dict[str, str]: Dict of (key, value) pairs corresponding to (alias, table name).
    """
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
    """Substitutes provided symbol for entire table expression in SQL query.

    Args:
        simple_sql_query (str): SQL query to redact.
        sql_table_expr (str): Table expression to replace.
        sub_symbol (str): Symbol with which to replace table expression.

    Returns:
        _type_: Redacted SQL query with symbol in place of table expression
    """
    idx = simple_sql_query.find(sql_table_expr)
    if idx < 0:
        print("[substitute_symbol_for_table] sql_table_expr not in simple_sql_query")
        return simple_sql_query

    return re.sub(sql_table_expr, sub_symbol, simple_sql_query)


def sql_table_expr_to_pandas(sql_table_expr: str) -> str:
    """TODO

    Args:
        sql_table_expr (str): _description_

    Returns:
        str: _description_
    """
    return ""
