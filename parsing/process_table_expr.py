from typing import Dict, List, Tuple
from clean_query import remove_consecutive_spaces
from helpers import get_next_token_idx, get_cur_token, get_prev_token, get_next_token, extract_table_column, get_first_token
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
        cur_token = get_cur_token(simple_sql_query, idx)
        if cur_token == "JOIN":
            idx = get_next_token_idx(simple_sql_query, idx)
            idx = get_next_token_idx(simple_sql_query, idx)
        elif cur_token == "AS":
            idx = get_next_token_idx(simple_sql_query, idx)
            idx = get_next_token_idx(simple_sql_query, idx)
        elif cur_token == "ON" or cur_token == "AND":
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


def extract_table_aliases_from_table_expr(sql_table_expr: str) -> Dict[str, str]:
    """Extracts AS aliases for tables in table expression.

    Args:
        sql_table_expr (str): SQL table expression from which to create alias table.

    Returns:
        Dict[str, str]: Dict of (key, value) pairs corresponding to (alias, table name).
    """
    table_alias_dict = dict()

    idx = 0
    while idx < len(sql_table_expr):
        cur_token = get_cur_token(sql_table_expr, idx)
        if cur_token == "AS":
            table_name = get_prev_token(sql_table_expr, idx)
            alias_name = get_next_token(sql_table_expr, idx)
            table_alias_dict.setdefault(alias_name, table_name)
            idx = get_next_token_idx(sql_table_expr, idx)
            idx = get_next_token_idx(sql_table_expr, idx)
        else:
            idx = get_next_token_idx(sql_table_expr, idx)

    return table_alias_dict


def remove_table_aliases(sql_table_expr: str) -> str:
    """Removes AS aliases for tables in table expression.

    Args:
        sql_table_expr (str): SQL table expression from which to create alias table.

    Returns:
        str: Redacted table expression, with just the aliases.
    """

    idx = 0
    aliased_sql_table_expr = ""
    while idx < len(sql_table_expr):
        cur_token = get_cur_token(sql_table_expr, idx)
        next_token = get_next_token(sql_table_expr, idx)
        if next_token == "AS":
            idx = get_next_token_idx(sql_table_expr, idx)
            idx = get_next_token_idx(sql_table_expr, idx)
        else:
            aliased_sql_table_expr += cur_token + " "
            idx = get_next_token_idx(sql_table_expr, idx)

    return remove_consecutive_spaces(aliased_sql_table_expr)


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


def extract_join_segments(aliased_sql_table_expr: str) -> List[str]:
    """Extracts JOIN segments.

    Args:
        aliased_sql_table_expr (str): SQL table expression with aliases only.

    Returns:
        List[str]: Extract tokens between JOINs.
    """
    # TODO: replace find JOIN with find(token) for list of tokens
    join_segments = []
    idx = 0
    while idx < len(aliased_sql_table_expr):
        next_idx = aliased_sql_table_expr.find("JOIN", idx)
        if next_idx < 0:
            segment = aliased_sql_table_expr[idx:]
            segment = remove_consecutive_spaces(segment)
            join_segments.append(segment)
            return join_segments

        segment = aliased_sql_table_expr[idx:next_idx]
        segment = remove_consecutive_spaces(segment)
        join_segments.append(segment)
        idx = next_idx + len("JOIN")

    return join_segments


# TODO: extract ONs for each segment
def extract_on_cols_from_join_segment(join_segment: str) -> List[Tuple[str, str]]:
    on_cols = []
    idx = 0
    while idx < len(join_segment):
        cur_token = get_cur_token(join_segment, idx)
        if cur_token == "ON":
            idx = get_next_token_idx(join_segment, idx)
            left_finish_idx = join_segment.find("=", idx)
            if left_finish_idx < 0:
                print("[extract_on_from_join_segment] left_finish_idx < 0")
                return on_cols

            left_col = join_segment[idx:left_finish_idx]
            left_col = remove_consecutive_spaces(left_col)
            idx = left_finish_idx + 1
            idx = get_next_token_idx(join_segment, idx)
            right_finish_idx = join_segment.find(" ", idx)
            if right_finish_idx < 0:
                right_finish_idx = len(join_segment)

            right_col = join_segment[idx:right_finish_idx]
            right_col = remove_consecutive_spaces(right_col)
            on_cols.append((left_col, right_col))
            idx = get_next_token_idx(join_segment, right_finish_idx)
        else:
            idx = get_next_token_idx(join_segment, idx)

    return on_cols


def join_two_tables(t1: str, t2: str, on_cols: List[Tuple[str, str]]) -> str:
    # TODO: different types of JOINs
    if len(on_cols) == 0:
        return f"pd.merge({t1}, {t2})"

    left_on = []
    right_on = []
    for l_on_col, r_on_col in on_cols:
        # TODO: clean this up?
        left_on.append(extract_table_column(l_on_col))
        right_on.append(extract_table_column(r_on_col))
    return f"pd.merge({t1}, {t2}, left_on={left_on}, right_on={right_on})"


def sql_table_expr_join_segments_to_snippets(join_segments: List[Tuple[str, str]], left_symbol: str, right_idx: int, pandas_snippets: List[str], get_symbol) -> str:
    if right_idx >= len(join_segments):
        return left_symbol

    right_segment = join_segments[right_idx]
    on_cols = extract_on_cols_from_join_segment(right_segment)
    joined_table_expr = join_two_tables(
        left_symbol, get_first_token(right_segment), on_cols)
    new_symbol = get_symbol()
    snippet = f"{new_symbol} = {joined_table_expr}"
    pandas_snippets.append(snippet)

    return sql_table_expr_join_segments_to_snippets(
        join_segments, new_symbol, right_idx+1, pandas_snippets, get_symbol)


def sql_table_expr_to_pandas_snippets(table_expr_symbol: str, aliased_sql_table_expr: str, get_symbol) -> List[str]:
    """Given aliased SQL table expression (i.e. JOIN, ON, AND), return code snippets corresponding to pandas conversion.

    Args:
        aliased_sql_table_expr (str): Simple table expression (no AS aliases, only symbols).

    Returns:
        List[str]: List of pandas snippets corresponding to aliased_sql_table_expr.
    """
    join_segments = extract_join_segments(aliased_sql_table_expr)

    if len(join_segments) == 0:
        return [join_segments[0]]

    pandas_snippets = []
    final_symbol = sql_table_expr_join_segments_to_snippets(
        join_segments, join_segments[0], 1, pandas_snippets, get_symbol)

    pandas_snippets.append(f"{table_expr_symbol} = {final_symbol}")

    return pandas_snippets
