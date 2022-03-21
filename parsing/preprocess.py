from turtle import right
from helpers import trim_front_and_back, find_closing_parenthesis, is_next_token_select
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


#
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
        return ProcessedSQLQueryNode(
            node_type=ProcessedSQLQueryNodeType.LEAF, processed_query=sql_query, pandas_query=sql2pandas(sql_query), left_node=None, right_node=None
        )

    print(subquery)
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
            node_type=query_type, processed_query=None, pandas_query=None, left_node=left_node, right_node=right_node)

        # root_node.dump_processed_sql_tree()
        return root_node

    left_query = sql_query[0:idx-len(query_type.value)]
    left_node = preprocess_sql_query_into_root_node(left_query, tree_header)

    right_node = preprocess_sql_query_into_root_node(subquery, tree_header)

    root_node = ProcessedSQLQueryNode(
        node_type=query_type, processed_query=None, pandas_query=None, left_node=left_node, right_node=right_node)

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

    return ProcessedSQLQueryNode(
        node_type=ProcessedSQLQueryNodeType.LEAF, processed_query=sql_query, pandas_query=sql2pandas(sql_query), left_node=None, right_node=None
    )


def preprocess_sql_query(sql_query: str) -> ProcessedSQLQueryTree:
    tree = ProcessedSQLQueryTree()

    root_node = preprocess_sql_query_into_root_node(sql_query, tree)

    tree.reset_root_node(root_node)
    return tree
