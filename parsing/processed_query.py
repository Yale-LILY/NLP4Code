from enum import Enum
from typing import Any, Dict, List, Union

from process_table_expr import extract_table_aliases_from_table_expr, remove_table_aliases, sql_table_expr_to_pandas_snippets


class ProcessedSQLQueryNodeType(Enum):
    LEAF = "LEAF"
    NESTED_SELECT = "(SELECT"
    INTERSECT = "INTERSECT "
    UNION = "UNION "
    EXCEPT = "EXCEPT "
    SUBTRACT = " - "


def dump_dict(dict_obj, indent=4):
    if dict_obj == None:
        print("None")
    else:
        indent_spaces = indent * " "
        for key in dict_obj.keys():
            print(f"{indent_spaces}{key}: {dict_obj[key]}")


class ProcessedSQLTableExpr:
    def __init__(
        self,
        orig_table_expr: str,
        table_expr_symbol_key: str,
        get_symbol,
    ):
        self.orig_table_expr = orig_table_expr
        aliased_table_expr = remove_table_aliases(orig_table_expr)
        self.aliased_table_expr = aliased_table_expr
        self.table_expr_symbol_key = table_expr_symbol_key
        self.table_aliases = extract_table_aliases_from_table_expr(
            orig_table_expr)
        self.table_expr_pandas_snippets = sql_table_expr_to_pandas_snippets(
            table_expr_symbol=table_expr_symbol_key, aliased_sql_table_expr=aliased_table_expr, get_symbol=get_symbol)

    def extract_table_aliases(self, code_snippets: List[str]):
        for key in self.table_aliases:
            code_snippets.append(f"{key} = {self.table_aliases[key]}")

    def dump_table_expr(self, indent=4):
        indent_spaces = indent * " "

        print(f"{indent_spaces}orig_table_expr: {self.orig_table_expr}")
        print(f"{indent_spaces}aliased_table_expr: {self.aliased_table_expr}")
        print(f"{indent_spaces}table_expr_symbol_key: {self.table_expr_symbol_key}")
        print(f"{indent_spaces}table_aliases:")
        dump_dict(self.table_aliases, indent=2*indent)
        print(
            f"{indent_spaces}table_expr_pandas_snippets: {self.table_expr_pandas_snippets}")


class ProcessedSQLQueryNode:
    """Tree node class for processed SQL queries.

    Attributes:
        node_type (ProcessedSQLQueryNodeType): Specifies node type.
        internal_symbol (str | None): Symbol in symbol_table.
            Unique symbol corresponding to SQL query rooted at node.
        processed_query (str | None): Stores sql2pandas-convertible SQL query if LEAF node.
        pandas_query (str | None): Stores sql2pandas-converted SQL query (corresponding to processed_query) if LEAF node.
        left_node (ProcessedSQLQueryNode | None): Left child node.
            Contains redacted SELECT query (redacted sub-query stored in right_node).
        right_node (ProcessedSQLQueryNode | None): Left child node.
            Connects smaller SELECT sub-query (fits into redacted portion of left_node query).
        external_symbol (str | None): Symbol in symbol_table.
            If processed_query is redacted with external_symbol substitute, can be used to lookup redacted sub_query.
            Only present if ancestor is NESTED_SELECT type.
    """

    def __init__(
            self,
            node_type: ProcessedSQLQueryNodeType,
            internal_symbol: str,
            sql_query: Union[str, None],
            sql_query_table_expr: Union[ProcessedSQLTableExpr, None],
            pandas_query: Union[str, None],
            left_node: Union[Dict[str, Any], None],
            right_node: Union[Dict[str, Any], None],
            external_symbol: Union[str, None] = None):
        self.node_type = node_type
        self.internal_symbol = internal_symbol
        self.sql_query = sql_query
        self.sql_query_table_expr = sql_query_table_expr
        self.pandas_query = pandas_query
        self.left_node = left_node
        self.right_node = right_node
        self.external_symbol = external_symbol

    def set_external_symbol(self, external_symbol: str):
        self.external_symbol = external_symbol

    def dump_processed_sql_tree(self):
        """Print contents of tree rooted at this node."""
        if not self.left_node == None:
            self.left_node.dump_processed_sql_tree()

        print(f"node_type: {self.node_type}")
        print(f"internal_symbol: {self.internal_symbol}")

        if self.node_type == ProcessedSQLQueryNodeType.LEAF:
            print(f"processed_query: {self.sql_query}")
            print("sql_query_table_expr:")
            self.sql_query_table_expr.dump_table_expr()
            print(f"pandas_query: {self.pandas_query}")
            print(f"external_symbol: {self.external_symbol}")

        print()

        if not self.right_node == None:
            self.right_node.dump_processed_sql_tree()


class ProcessedSQLQueryTree:
    """Represents ProcessedSQLQueryTree header node, with symbol table.

    Attributes:
        root_node (ProcessedSQLQueryNode): Root node of SQL query tree.
        symbol_table (Dict[str, Tuple[str, ProcessedSQLQueryNode]]): Dict with all symbols, mapped to (query, node) tuple.
        symbol_count (int): Number of symbols in symbol_table.
    """

    def __init__(self, root_node: Union[ProcessedSQLQueryNode, None] = None):
        self.root_node = root_node
        self.symbol_table = dict()
        self.symbol_count = 0

    def get_symbol_key(self):
        """Generate symbol key based on number of symbols currently in tree."""
        symbol_key = f"symbol_{self.symbol_count}"
        self.symbol_count += 1
        return symbol_key

    def add_key_value_to_symbol_table(self, symbol_key: str, query_str: str, tree_node: ProcessedSQLQueryNode):
        """Add new (key, value) to tree symbol_table.

        Args:
            symbol_key (str): Symbol key. Preferably generated by get_symbol_key().
            query_str (str): Query for which key substitutes.
            tree_node (ProcessedSQLQueryNode): Node in tree at which query_str is rooted.
        """
        self.symbol_table[symbol_key] = (query_str, tree_node)

    def reset_root_node(self, new_root_node: ProcessedSQLQueryNode):
        self.root_node = new_root_node

    def dump_tree(self):
        """Print symbol table/count, as well as tree contents."""
        print("-------- ProcessedSQLQueryTree --------\n")
        print("-------- symbol_table --------")
        for key in self.symbol_table.keys():
            (query_str, tree_node) = self.symbol_table[key]
            print(f"{key}: {query_str} (rooted at node {tree_node})")

        print("\n-------- symbol_count --------")
        print(str(self.symbol_count))

        print("\n-------- Processed SQL query tree: --------\n")
        self.root_node.dump_processed_sql_tree()
