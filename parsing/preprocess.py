from typing import Dict, List
from helpers import trim_front_and_back, find_closing_parenthesis, is_next_token_select, get_cur_token, get_next_token, get_next_token_idx, get_prev_token
from processed_query import ProcessedSQLQueryNode, ProcessedSQLQueryNodeType, ProcessedSQLQueryTree
from sql2pandas import sql2pandas
import re


# TODO:
# - rewrite sql2pandas API to accept ProcessedSQLQueryTree
# - INTERSECT, UNION, EXCEPT
# - add complex tables to their own symbol table? (handle JOIN and table AS aliases)
#   - JOIN (fully joined table, i.e. t1 JOIN t2 ON t1.id = t2.id JOIN t3 [...])
#       included in symbol table as SYMBOL_N = "t1 JOIN t2 ON t1.id = t2.id JOIN t3 [...]"
#   - remove all AS table aliases for all leaves
# - add way to link external/internal symbols? (combine separate pandas_query's from leaves to get one giant pandas query)


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


def extract_select_subquery(sql_query: str, query_type: ProcessedSQLQueryNodeType):
    query_type_token = query_type.value

    start_idx = sql_query.find(query_type_token)
    if start_idx < 0:
        return None

    start_idx += 1 if query_type == ProcessedSQLQueryNodeType.NESTED_SELECT else len(
        query_type_token)
    if not is_next_token_select(sql_query[start_idx:]):
        return None

    # TODO: better finishing idx
    finish_idx = len(sql_query)
    if query_type == ProcessedSQLQueryNodeType.NESTED_SELECT:
        finish_idx = find_closing_parenthesis(sql_query, start_idx)
        if finish_idx == -1:
            print("[handle_nested_select] parenthesis imbalance detected: " + sql_query)
            return None

    nested_query = sql_query[start_idx:finish_idx]
    return nested_query


def handle_select_subquery(sql_query: str, tree_header: ProcessedSQLQueryTree, query_type: ProcessedSQLQueryNodeType) -> ProcessedSQLQueryNode:
    subquery = extract_select_subquery(sql_query, query_type)
    if subquery == None:
        sql_table = extract_table_expr_from_query(sql_query)
        table_aliases = extract_table_aliases(sql_table)
        table_symbol = dict()
        table_symbol_key = tree_header.get_symbol_key()
        tree_header.increment_symbol_count()
        table_symbol.setdefault(table_symbol_key, sql_table)

        final_sql_query = substitute_symbol_for_table_expr(
            sql_query, sql_table, table_symbol_key)

        return ProcessedSQLQueryNode(
            node_type=ProcessedSQLQueryNodeType.LEAF,
            sql_query=final_sql_query,
            sql_query_table_symbol=table_symbol,
            sql_query_table_aliases=table_aliases,
            pandas_query=sql2pandas(final_sql_query),
            left_node=None,
            right_node=None
        )

    idx = sql_query.find(subquery)
    if idx < 0:
        print("[preprocess.py] ERROR: could not find subquery in sql_query")
        return sql_query

    symbol_key = tree_header.get_symbol_key()

    if query_type == ProcessedSQLQueryNodeType.NESTED_SELECT:
        left_query = sql_query[0:idx] + \
            symbol_key + sql_query[idx+len(subquery):]

        left_node = preprocess_sql_query_into_root_node(
            left_query, tree_header)
        left_node.set_external_symbol(symbol_key)

        right_node = preprocess_sql_query_into_root_node(
            subquery, tree_header)
        right_node.set_internal_symbol(symbol_key)

        tree_header.add_key_value_to_symbol_table(
            symbol_key, subquery, right_node)

        root_node = ProcessedSQLQueryNode(
            node_type=query_type,
            sql_query=None,
            sql_query_table_symbol=None,
            sql_query_table_aliases=None,
            pandas_query=None,
            left_node=left_node,
            right_node=right_node
        )

        # root_node.dump_processed_sql_tree()
        return root_node

    left_query = sql_query[0:idx-len(query_type.value)]
    left_node = preprocess_sql_query_into_root_node(left_query, tree_header)

    right_node = preprocess_sql_query_into_root_node(subquery, tree_header)

    root_node = ProcessedSQLQueryNode(
        node_type=query_type, sql_query=None, sql_query_table_symbol=None, sql_query_table_aliases=None, pandas_query=None, left_node=left_node, right_node=right_node)

    # root_node.dump_processed_sql_tree()
    return root_node


# sql2pandas requires single quotes in SQL queries
def replace_quotes(sql_query):
    return sql_query.replace("\"", "\'")


# Remove extra spaces
def remove_consecutive_spaces(sql_query):
    sql_query = sql_query.strip()
    sql_query = re.sub(r"\s+", " ", sql_query)
    sql_query = re.sub(r"\( ", "(", sql_query)
    return sql_query


# Add semi-colon at end of SQL query for consistency
def add_semicolon(sql_query):
    return sql_query if sql_query[-1:] == ";" else sql_query + ";"


# Basic string preprocessing/cleanup for SQL queries
def basic_string_preprocess(sql_query):
    sql_query = replace_quotes(sql_query)
    sql_query = remove_consecutive_spaces(sql_query)
    # TODO: ensure balance for front/back parentheses
    sql_query = trim_front_and_back(sql_query, "(", ")")

    sql_query = add_semicolon(sql_query)
    return sql_query


def preprocess_sql_query_into_root_node(sql_query: str, tree_header: ProcessedSQLQueryTree) -> ProcessedSQLQueryNode:
    sql_query = basic_string_preprocess(sql_query)

    for query_type in ProcessedSQLQueryNodeType:
        if not extract_select_subquery(sql_query, query_type) == None:
            return handle_select_subquery(sql_query, tree_header, query_type)

    sql_table = extract_table_expr_from_query(sql_query)
    table_aliases = extract_table_aliases(sql_table)
    table_symbol = dict()
    table_symbol_key = tree_header.get_symbol_key()
    tree_header.increment_symbol_count()
    table_symbol.setdefault(table_symbol_key, sql_table)

    final_sql_query = substitute_symbol_for_table_expr(
        sql_query, sql_table, table_symbol_key)

    return ProcessedSQLQueryNode(
        node_type=ProcessedSQLQueryNodeType.LEAF,
        sql_query=final_sql_query,
        sql_query_table_symbol=table_symbol,
        sql_query_table_aliases=table_aliases,
        pandas_query=sql2pandas(final_sql_query),
        left_node=None,
        right_node=None
    )


def preprocess_sql_query(sql_query: str) -> ProcessedSQLQueryTree:
    tree = ProcessedSQLQueryTree()

    root_node = preprocess_sql_query_into_root_node(sql_query, tree)

    tree.reset_root_node(root_node)
    return tree


def get_pandas_code_snippet_from_tree_dfs(sql_query_node: ProcessedSQLQueryNode, code_snippets: List[str]):
    if sql_query_node == None:
        return

    if sql_query_node.node_type == ProcessedSQLQueryNodeType.LEAF:
        for alias_symbol in sql_query_node.sql_query_table_aliases.keys():
            code_snippets.append(alias_symbol + " = " +
                                 sql_query_node.sql_query_table_aliases[alias_symbol])
        for table_sub in sql_query_node.sql_query_table_symbol.keys():
            code_snippets.append(table_sub + " = " +
                                 sql_query_node.sql_query_table_symbol[table_sub])
        code_snippets.append(sql_query_node.pandas_query)
        return

    return


def get_pandas_code_snippet_from_tree(sql_query_tree: ProcessedSQLQueryTree):
    code_snippets = list()
    get_pandas_code_snippet_from_tree_dfs(
        sql_query_tree.root_node, code_snippets)
    return code_snippets
