from processed_query import ProcessedSQLQueryNodeType, ProcessedSQLQueryNode


def symbols_to_pandas(s1: str, s2: str, node_type: ProcessedSQLQueryNodeType) -> str:
    """Given two symbols and SQL operation type on them, return corresponding pandas snippet."""
    if node_type == ProcessedSQLQueryNodeType.UNION:
        return "pd.concat([" + s1 + ", " + s2 + "]).drop_duplicates()"

    if node_type == ProcessedSQLQueryNodeType.NESTED_SELECT:
        return s1

    # TODO: how to do INTERSECT/EXCEPT?
    return s1 + " " + node_type.name + " " + s2


def extract_pandas_code_snippet_from_node(sql_query_node: ProcessedSQLQueryNode) -> str:
    """Extract pandas snippet representing one node in SQL query decomposition tree, using internal symbol.

    Args:
        sql_query_node (ProcessedSQLQueryNode): Node for which to generate snippet

    Returns:
        str: One pandas code snippet of the form "symbol = f(left_symbol, r_symbol)".
    """
    symbol = sql_query_node.internal_symbol
    if sql_query_node.node_type == ProcessedSQLQueryNodeType.LEAF:
        # Pandas query
        return symbol + " = " + sql_query_node.pandas_query

    left_symbol, right_symbol = sql_query_node.left_node.internal_symbol, sql_query_node.right_node.internal_symbol
    return symbol + " = " + symbols_to_pandas(left_symbol, right_symbol, sql_query_node.node_type)
