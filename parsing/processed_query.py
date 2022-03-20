from enum import Enum
from typing import Dict, Union


class ProcessedSQLQueryNodeType(Enum):
    LEAF = 'LEAF'
    NESTED_SELECT = 'NESTED_SELECT'
    INTERSECT = 'INTERSECT'
    UNION = 'UNION'
    EXCEPT = 'EXCEPT'


class ProcessedSQLQueryNode:
    def __init__(
            self,
            node_type: ProcessedSQLQueryNodeType,
            processed_query: Union[str, None],
            left_node: Union[Dict[str, any], None],
            right_node: Union[Dict[str, any], None]):
        self.original_query = ""
        self.query_symbol_table = None
        self.processed_query = processed_query
        self.node_type = node_type
        self.left_node = left_node
        self.right_node = right_node

    def dump_processed_sql_tree(self):
        if not self.left_node == None:
            self.left_node.dump_processed_sql_tree()

        print(self.node_type)
        print(str(self.processed_query) + "\n")

        if not self.right_node == None:
            self.right_node.dump_processed_sql_tree()


# # Tree-like structure
# # Leaves are S2P-convertible queries
# class ProcessedSQLQuery:
#     def __init__(self, raw_query: str):
#         self.raw_query = raw_query
#         self.processed_query = preprocess_sql_query(raw_query)

#     def dump_processed_sql_query(self):
#         print(self.raw_query)
#         print(self.processed_query)
