from typing import Dict, List, Union
from helpers import find_closing_parenthesis, is_next_token_select
from clean_query import basic_clean_query
from node_to_pandas_snippet import extract_pandas_code_snippet_from_node
from process_table_expr import extract_table_expr_from_query, substitute_symbol_for_table_expr
from processed_query import ProcessedSQLQueryNode, ProcessedSQLQueryNodeType, ProcessedSQLQueryTree, ProcessedSQLTableExpr
from sql2pandas import sql2pandas
import re


# TODO:
# - convert table_expr to pandas (https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html)
# - handle RIGHT/LEFT OUTER/INNER/FULL JOIN
# - handle UNION ALL
# - remove original table name from table_expr (or replace all instances of AS alias with orig table name)
# - return None tree if SQL query generates error (error in sql2pandas, parenthesis imbalance, etc)


def extract_select_subquery(sql_query: str, query_type: ProcessedSQLQueryNodeType) -> Union[str, None]:
    """Finds and extracts SELECT subquery from complex SQL query.

    Args:
        sql_query (str): SQL query from which to find and extract SELECT subquery.
        query_type (ProcessedSQLQueryNodeType): Type of SELECT subquery.

    Returns:
        Union[str, None]: Extracted subquery (starting from SELECT) if found, None else.
    """
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

    extracted_subquery = sql_query[start_idx:finish_idx]
    return extracted_subquery


def convert_query_to_tree_node(sql_query: str, internal_symbol: str, tree_header: ProcessedSQLQueryTree) -> ProcessedSQLQueryNode:
    """If there are SELECT subqueries to extract, extracts one layer of SELECT subquery into a tree,
    with recursion to generate subtrees on remaining layers.

    Args:
        sql_query (str): Full SQL query string to decompose.
        tree_header (ProcessedSQLQueryTree): Tree header.

    Returns:
        ProcessedSQLQueryNode: Tree node rooted at tree containing decomposed SQL query.
    """

    sql_query = basic_clean_query(sql_query)

    query_type, subquery = None, None
    for find_query_type in ProcessedSQLQueryNodeType:
        found_subquery = extract_select_subquery(sql_query, find_query_type)
        if found_subquery != None:
            query_type = find_query_type
            subquery = found_subquery
            break

    # Base case: LEAF node
    if query_type == None or subquery == None:
        table_expr_str = extract_table_expr_from_query(sql_query)
        table_expr_symbol_key = tree_header.get_symbol_key()
        table_expr = ProcessedSQLTableExpr(
            orig_table_expr=table_expr_str, table_expr_symbol_key=table_expr_symbol_key)

        final_sql_query = substitute_symbol_for_table_expr(
            sql_query, table_expr_str, table_expr_symbol_key)

        leaf_node = ProcessedSQLQueryNode(
            node_type=ProcessedSQLQueryNodeType.LEAF,
            internal_symbol=internal_symbol,
            sql_query=final_sql_query,
            sql_query_table_expr=table_expr,
            pandas_query=sql2pandas(final_sql_query),
            left_node=None,
            right_node=None
        )

        tree_header.add_key_value_to_symbol_table(
            internal_symbol, sql_query, leaf_node)
        return leaf_node

    idx = sql_query.find(subquery)
    if idx < 0:
        print("[preprocess.py] ERROR: could not find subquery in sql_query")
        return sql_query

    left_symbol_key = tree_header.get_symbol_key()
    right_symbol_key = tree_header.get_symbol_key()

    if query_type == ProcessedSQLQueryNodeType.NESTED_SELECT:
        left_query = sql_query[0:idx] + \
            right_symbol_key + sql_query[idx+len(subquery):]

        left_node = convert_query_to_tree_node(
            left_query, left_symbol_key, tree_header)
        left_node.set_external_symbol(right_symbol_key)

        right_node = convert_query_to_tree_node(
            subquery, right_symbol_key, tree_header)

        root_node = ProcessedSQLQueryNode(
            node_type=query_type,
            internal_symbol=internal_symbol,
            sql_query=None,
            sql_query_table_expr=None,
            pandas_query=None,
            left_node=left_node,
            right_node=right_node
        )

        tree_header.add_key_value_to_symbol_table(
            internal_symbol, sql_query, root_node)
        return root_node

    left_query = sql_query[0:idx-len(query_type.value)]
    left_node = convert_query_to_tree_node(
        left_query, left_symbol_key, tree_header)
    right_node = convert_query_to_tree_node(
        subquery, right_symbol_key, tree_header)

    root_node = ProcessedSQLQueryNode(
        node_type=query_type,
        internal_symbol=internal_symbol,
        sql_query=None,
        sql_query_table_expr=None,
        pandas_query=None,
        left_node=left_node,
        right_node=right_node
    )

    tree_header.add_key_value_to_symbol_table(
        internal_symbol, sql_query, root_node)
    return root_node


def preprocess_sql_query(sql_query: str) -> ProcessedSQLQueryTree:
    """Processes SQL query string into ProcessedSQLQueryTree.

    Args:
        sql_query (str): SQL query string to decompose.

    Returns:
        ProcessedSQLQueryTree: SQL query string tree decomposition.
    """
    tree = ProcessedSQLQueryTree()

    root_symbol = tree.get_symbol_key()
    root_node = convert_query_to_tree_node(sql_query, root_symbol, tree)

    tree.reset_root_node(root_node)
    return tree


def extract_pandas_table_aliases_dfs(node: ProcessedSQLQueryNode, code_snippets: List[str]):
    """Helper function to extract all table aliases into executable Python code snippets"""
    if node == None:
        return

    if node.node_type == ProcessedSQLQueryNodeType.LEAF:
        node.sql_query_table_expr.extract_table_aliases(code_snippets)
        return

    extract_pandas_table_aliases_dfs(node.right_node, code_snippets)
    extract_pandas_table_aliases_dfs(node.left_node, code_snippets)


def extract_pandas_table_expr_symbols_dfs(node: ProcessedSQLQueryNode, code_snippets: List[str]):
    """Helper function to extract all table expression substitute symbols into executable Python code snippets"""
    if node == None:
        return

    if node.node_type == ProcessedSQLQueryNodeType.LEAF:
        table_expr = node.sql_query_table_expr
        code_snippets.append(table_expr.table_expr_symbol_key + " = " +
                             table_expr.aliased_table_expr)  # TODO: turn this into pandas
        return

    extract_pandas_table_expr_symbols_dfs(node.right_node, code_snippets)
    extract_pandas_table_expr_symbols_dfs(node.left_node, code_snippets)


def get_pandas_code_snippets_dfs(sql_query_node: ProcessedSQLQueryNode, code_snippets: List[str]):
    """Helper function for get_pandas_code_snippet_from_tree"""
    if sql_query_node == None:
        return

    # Postorder: right (nested) to left, then root
    get_pandas_code_snippets_dfs(
        sql_query_node.right_node, code_snippets)
    get_pandas_code_snippets_dfs(
        sql_query_node.left_node, code_snippets)

    code_snippets.append(extract_pandas_code_snippet_from_node(sql_query_node))


def get_pandas_code_snippet_from_tree(sql_query_tree: ProcessedSQLQueryTree) -> List[str]:
    """Generate list of executable pandas code from SQL query tree decomposition.

    Args:
        sql_query_tree (ProcessedSQLQueryTree): Tree from which to extract pandas code.

    Returns:
        List[str]: List of executable pandas statements, in order.
    """
    code_snippets = list()
    extract_pandas_table_aliases_dfs(
        sql_query_tree.root_node, code_snippets)
    extract_pandas_table_expr_symbols_dfs(
        sql_query_tree.root_node, code_snippets)
    get_pandas_code_snippets_dfs(sql_query_tree.root_node, code_snippets)

    # Temp: remove duplicate code snippets (from repeated tables in subqueries)
    # TODO: move table aliases to tree metainformation
    return list(dict.fromkeys(code_snippets))


def check_processed_sql_tree_dfs(node: ProcessedSQLQueryNode) -> Union[str, None]:
    """Helper function for check_processed_sql_tree."""
    if node == None:
        return None

    if node.node_type == ProcessedSQLQueryNodeType.LEAF:
        assert(node.left_node == None and node.right_node == None)
        assert(node.sql_query != None)
        assert(node.sql_query_table_expr != None)
        assert(node.pandas_query != None)
        # assert(node.pandas_query.find("Error:") < 0)
        if node.pandas_query.find("Error:") >= 0:
            return node.sql_query + " -> " + node.pandas_query
        return None

    assert(node.left_node != None and node.right_node != None)
    assert(node.sql_query == None)
    assert(node.sql_query_table_expr == None)
    assert(node.pandas_query == None)

    left_res = check_processed_sql_tree_dfs(node.left_node)
    if left_res != None:
        return left_res

    right_res = check_processed_sql_tree_dfs(node.right_node)
    if right_res != None:
        return right_res

    return None


def check_processed_sql_tree(sql_query_tree: ProcessedSQLQueryTree) -> Union[str, None]:
    """Checks validity of ProcessedSQLQueryTree.

    Ensure LEAF and non-LEAF nodes contain required fields (assertions fail if not),
    and that pandas queries for LEAF nodes were generated without errors.

    Args:
        sql_query_tree (ProcessedSQLQueryTree): Tree to check validity.

    Returns:
        Union[str, None]: Error string if tree is invalid, or None if tree is valid.
    """
    return check_processed_sql_tree_dfs(sql_query_tree.root_node)
