from processed_query import ProcessedSQLQueryNodeType, ProcessedSQLQueryNode


# TODO:
# - INTERSECT/EXCEPT
# - UNION ALL, other unsupported SQL keywords?


def symbols_to_pandas(s1: str, s2: str, node_type: ProcessedSQLQueryNodeType) -> str:
    """Given two symbols and SQL operation type on them, return corresponding pandas snippet."""

    if node_type == ProcessedSQLQueryNodeType.NESTED_SELECT:
        return s1

    if node_type == ProcessedSQLQueryNodeType.UNION:
        return f"pd.concat([{s1}, {s2}]).drop_duplicates()"

    # TODO: how to do INTERSECT/EXCEPT?
    return f"{s1} {node_type.name} {s2}"


def extract_pandas_code_snippet_from_node(sql_query_node: ProcessedSQLQueryNode) -> str:
    """Extract pandas snippet representing one node in SQL query decomposition tree, using internal symbol.

    Args:
        sql_query_node (ProcessedSQLQueryNode): Node for which to generate snippet

    Returns:
        str: One pandas code snippet of the form "symbol = f(left_symbol, r_symbol)".
    """
    symbol = sql_query_node.internal_symbol
    if sql_query_node.node_type == ProcessedSQLQueryNodeType.LEAF:
        return f"{symbol} = {sql_query_node.pandas_query}"

    left_symbol = sql_query_node.left_node.internal_symbol
    right_symbol = sql_query_node.right_node.internal_symbol
    return f"{symbol} = {symbols_to_pandas(left_symbol, right_symbol, sql_query_node.node_type)}"
