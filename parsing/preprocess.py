from typing import Dict, List, Union
from helpers import find_closing_parenthesis, is_next_token_select
from clean_query import basic_clean_query
from process_table_expr import extract_table_expr_from_query, substitute_symbol_for_table_expr
from processed_query import ProcessedSQLQueryNode, ProcessedSQLQueryNodeType, ProcessedSQLQueryTree, ProcessedSQLTableExpr
from sql2pandas import sql2pandas
import re


# TODO:
# - convert table_expr to pandas (https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html)
# - handle RIGHT/LEFT OUTER/INNER/FULL JOIN
# - handle UNION ALL
# - remove original table name from table_expr (or replace all instances of AS alias with orig table name)


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


def convert_query_to_tree_node(sql_query: str, tree_header: ProcessedSQLQueryTree) -> ProcessedSQLQueryNode:
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

        return ProcessedSQLQueryNode(
            node_type=ProcessedSQLQueryNodeType.LEAF,
            sql_query=final_sql_query,
            sql_query_table_expr=table_expr,
            pandas_query=sql2pandas(final_sql_query),
            left_node=None,
            right_node=None
        )

    idx = sql_query.find(subquery)
    if idx < 0:
        print("[preprocess.py] ERROR: could not find subquery in sql_query")
        return sql_query

    if query_type == ProcessedSQLQueryNodeType.NESTED_SELECT:
        symbol_key = tree_header.get_symbol_key()

        left_query = sql_query[0:idx] + \
            symbol_key + sql_query[idx+len(subquery):]

        left_node = convert_query_to_tree_node(
            left_query, tree_header)
        left_node.set_external_symbol(symbol_key)

        right_node = convert_query_to_tree_node(
            subquery, tree_header)
        right_node.set_internal_symbol(symbol_key)

        tree_header.add_key_value_to_symbol_table(
            symbol_key, subquery, right_node)

        root_node = ProcessedSQLQueryNode(
            node_type=query_type,
            sql_query=None,
            sql_query_table_expr=None,
            pandas_query=None,
            left_node=left_node,
            right_node=right_node
        )

        # root_node.dump_processed_sql_tree()
        return root_node

    left_query = sql_query[0:idx-len(query_type.value)]
    left_node = convert_query_to_tree_node(left_query, tree_header)

    right_node = convert_query_to_tree_node(subquery, tree_header)

    root_node = ProcessedSQLQueryNode(
        node_type=query_type,
        sql_query=None,
        sql_query_table_expr=None,
        pandas_query=None,
        left_node=left_node,
        right_node=right_node
    )

    # root_node.dump_processed_sql_tree()
    return root_node


def preprocess_sql_query(sql_query: str) -> ProcessedSQLQueryTree:
    """Processes SQL query string into ProcessedSQLQueryTree.

    Args:
        sql_query (str): SQL query string to decompose.

    Returns:
        ProcessedSQLQueryTree: SQL query string tree decomposition.
    """
    tree = ProcessedSQLQueryTree()

    root_node = convert_query_to_tree_node(sql_query, tree)

    tree.reset_root_node(root_node)
    return tree


def get_pandas_code_snippet_from_tree_dfs(sql_query_node: ProcessedSQLQueryNode, code_snippets: List[str]):
    """Helper function for get_pandas_code_snippet_from_tree"""
    if sql_query_node == None:
        return

    if sql_query_node.node_type == ProcessedSQLQueryNodeType.LEAF:
        table_expr = sql_query_node.sql_query_table_expr

        # Table aliases
        # TODO: move this to table_expr class?
        table_aliases = table_expr.table_aliases
        for alias_symbol in table_aliases.keys():
            code_snippets.append(alias_symbol + " = " +
                                 table_aliases[alias_symbol])

        # Table expr symbol key
        code_snippets.append(table_expr.table_expr_symbol_key + " = " +
                             table_expr.orig_table_expr + " -> " + table_expr.pandas_table_expr)

        # Pandas query
        if sql_query_node.internal_symbol != None:
            code_snippets.append(
                sql_query_node.internal_symbol + " = " + sql_query_node.pandas_query)
        else:
            code_snippets.append(sql_query_node.pandas_query)

    # Postorder: right (nested) to left
    get_pandas_code_snippet_from_tree_dfs(
        sql_query_node.right_node, code_snippets)
    get_pandas_code_snippet_from_tree_dfs(
        sql_query_node.left_node, code_snippets)


def get_pandas_code_snippet_from_tree(sql_query_tree: ProcessedSQLQueryTree) -> List[str]:
    """Generate list of executable pandas code from SQL query tree decomposition.

    Args:
        sql_query_tree (ProcessedSQLQueryTree): Tree from which to extract pandas code.

    Returns:
        List[str]: List of executable pandas statements, in order.
    """
    code_snippets = list()
    get_pandas_code_snippet_from_tree_dfs(
        sql_query_tree.root_node, code_snippets)
    return code_snippets


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
