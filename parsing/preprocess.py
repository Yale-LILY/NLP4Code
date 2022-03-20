from turtle import right
from helpers import trim_front_and_back, find_closing_parenthesis, is_next_token_select
from processed_query import ProcessedSQLQueryNode, ProcessedSQLQueryNodeType, ProcessedSQLQueryTree
from sql2pandas import sql2pandas
import re

# For sql2pandas(query), the function first needs to call `preprocess` on query. (query is just a SQL command)
# If there is a JOIN in the query, replace all JOINs with just one symbol to represent the JOINed table,
# then substitute the JOIN back in after turning the JOIN to pandas
# If there is an INTERSECT/UNION/EXCEPT in the query, split into two SELECTS and pass up separately.
# If there is a nested SELECT, replace that with a symbol, pass both SELECTs up (but with a symbol for the first one),
# then plug in the pandas for the nested SELECT into the first one where the symbol is the pandas for the second SELECT

# To handle multiple table aliases, remove AS and token before

# Remove SQL syntax not supported by sql2pandas
# - `AS` aliases on multiple tables (see JOIN)
# - TODO: consider removing all `AS` aliases and using symbol table?
# - `JOIN` (fully joined table, i.e. `t1 JOIN t2 ON t1.id = t2.id JOIN t3 [...]` included in symbol table)
# - `INTERSECT`, `UNION`, `EXCEPT`
# - nested `SELECT`


# Extract nested SELECT query, with open/close parentheses
def extract_nested_select(sql_query: str):
    start_idx = sql_query.find("(")
    if start_idx < 0:
        return None

    start_idx += 1
    if not is_next_token_select(sql_query[start_idx:]):
        return None

    finish_idx = find_closing_parenthesis(sql_query, start_idx)

    if finish_idx == -1:
        print("[handle_nested_select] parenthesis imbalance detected: " + sql_query)
        return None

    nested_query = sql_query[start_idx:finish_idx]
    return nested_query


# Extract nested SELECT (only one layer) if it exists in `sql_query`
def handle_nested_select(sql_query: str, tree_header: ProcessedSQLQueryTree) -> ProcessedSQLQueryNode:
    nested_query = extract_nested_select(sql_query)
    if nested_query == None:
        return ProcessedSQLQueryNode(
            node_type=ProcessedSQLQueryNodeType.LEAF, processed_query=sql_query, pandas_query=sql2pandas(sql_query), left_node=None, right_node=None
        )

    idx = sql_query.find(nested_query)
    if idx < 0:
        print("[preprocess.py] ERROR: could not find nested_query in sql_query")
        return sql_query

    symbol_key = tree_header.get_key()
    left_query = sql_query[0:idx] + \
        symbol_key + sql_query[idx+len(nested_query):]

    left_node = preprocess_sql_query_into_root_node(left_query, tree_header)
    left_node.set_external_key(symbol_key)

    right_node = preprocess_sql_query_into_root_node(nested_query, tree_header)
    right_node.set_internal_key(symbol_key)

    tree_header.add_key_value_to_symbol_table(
        symbol_key, nested_query, right_node)

    root_node = ProcessedSQLQueryNode(
        node_type=ProcessedSQLQueryNodeType.NESTED_SELECT, processed_query=None, pandas_query=None, left_node=left_node, right_node=right_node)

    # root_node.dump_processed_sql_tree()
    return root_node


# sql2pandas requires single quotes in SQL queries
def replace_quotes(sql_query):
    return sql_query.replace("\"", "\'")


# Remove extra spaces
def remove_consecutive_spaces(sql_query):
    sql_query = sql_query.strip()
    sql_query = re.sub(r"\s+", " ", sql_query)
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
    return handle_nested_select(sql_query, tree_header)


def preprocess_sql_query(sql_query: str) -> ProcessedSQLQueryTree:
    tree = ProcessedSQLQueryTree()

    root_node = preprocess_sql_query_into_root_node(sql_query, tree)

    tree.reset_root_node(root_node)
    return tree
