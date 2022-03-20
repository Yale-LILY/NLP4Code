from enum import Enum
from typing import Dict, Union


class ProcessedSQLQueryNodeType(Enum):
    LEAF = 'LEAF'
    NESTED_SELECT = 'NESTED_SELECT'
    INTERSECT = 'INTERSECT'
    UNION = 'UNION'
    EXCEPT = 'EXCEPT'


# Node class for ProcessedSQLQueryTree
# Leaf nodes are sql2pandas-convertible
# Non-leaf nodes specify operations not supported by sql2pandas
class ProcessedSQLQueryNode:
    def __init__(
            self,
            node_type: ProcessedSQLQueryNodeType,
            processed_query: Union[str, None],
            left_node: Union[Dict[str, any], None],
            right_node: Union[Dict[str, any], None],
            l_to_r_key: Union[str, None] = None):
        self.original_query = ""
        self.query_symbol_table = None
        self.processed_query = processed_query
        self.node_type = node_type
        self.left_node = left_node
        self.right_node = right_node
        self.l_to_r_key = l_to_r_key

    def set_l_to_r_key(self, l_to_r_key: str):
        self.l_to_r_key = l_to_r_key

    def dump_processed_sql_tree(self):
        if not self.left_node == None:
            self.left_node.dump_processed_sql_tree()

        print("node_type: " + str(self.node_type))
        print("processed_query: " + str(self.processed_query))
        print("l_to_r_key: " + str(self.l_to_r_key) + "\n")

        if not self.right_node == None:
            self.right_node.dump_processed_sql_tree()


# Header node for ProcessedSQLQuery tree
# Symbol
class ProcessedSQLQueryTree:
    def __init__(self, root_node: Union[ProcessedSQLQueryNode, None] = None):
        self.root_node = root_node
        self.symbol_table = dict()
        self.symbol_count = 0

    def get_key(self):
        return "SYMBOL_" + str(self.symbol_count)

    def add_key_value_to_symbol_table(self, key: str, query_str: str, tree_node: ProcessedSQLQueryNode):
        self.symbol_table[key] = (query_str, tree_node)
        self.symbol_count += 1

    def reset_root_node(self, new_root_node: ProcessedSQLQueryNode):
        self.root_node = new_root_node

    def dump_tree(self):
        print("-------- ProcessedSQLQueryTree --------\n")
        print("symbol_table:")
        for key in self.symbol_table.keys():
            (query_str, _x) = self.symbol_table[key]
            print(key + ": " + query_str)

        print("\n")
        print("symbol_count: " + str(self.symbol_count))

        print("\n")
        self.root_node.dump_processed_sql_tree()
