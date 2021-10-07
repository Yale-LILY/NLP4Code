from typing import List, Dict

COM_STMTS = ['if_statement', 'for_statement', 'while_statement', 'try_statement', 'with_statement',
             'function_definition', 'class_definition']
PY_MODULES = ['module', 'block', 'decorated_definition']

def get_statements_from_code(code: str, parser, tolerate_errors: bool=False):
    parsed_tree = parser.parse(bytes(code, 'utf-8'))

    # do a dfs on the parsed tree to record all the simple statements
    target_stmts: List[Dict] = []
    node_stack = [parsed_tree.root_node]
    while len(node_stack) > 0:
        node = node_stack.pop()

        if (node.type.endswith('statement') or node.type in ['comment', 'decorator']) \
            and node.type not in COM_STMTS:
            # this is a simple statement or a comment, so we can add it to the list
            target_stmts.append({'type': node.type, 'start_point': node.start_point, 
                                 'end_point': node.end_point, 'end_byte': node.end_byte})
        elif node.type in COM_STMTS or node.type.endswith('clause'):
            # separate the header and the body by the ":" token
            children_types = [c.type for c in node.children]
            separator_idx = children_types.index(':')
            assert separator_idx != -1

            # start of the header is the starter of the complex stmt, end is the end of the ":" token
            target_stmts.append({'type': node.type+'_header', 'start_point': node.start_point, 
                                 'end_point': node.children[separator_idx].end_point, 
                                 'end_byte': node.children[separator_idx].end_byte})
            node_stack.extend(node.children[separator_idx+1:][::-1])
        elif node.type in PY_MODULES:
            node_stack.extend(node.children[::-1])
        elif node.type == 'ERROR':
            # err_code_line = code[:byte_idx_to_char_idx(node.end_byte, code)].split('\n')[-1]
            # print(f"failed to parse code: #########\n{err_code_line}\n#########")
            if tolerate_errors:
                continue
            else:
                # failed to parse tree, return None NOTE: previously returning [], but this will get 
                # confused with blank cells
                return None
        else:
            # other types, not sure what it contains, but assume it doesn't contain more statements
            print(f'unexpected node type: {node.type}')
            assert 'statement' not in node.sexp()

    return target_stmts