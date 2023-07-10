import json
import os
import math

from typing import List, Dict, Any
from csv import DictReader
from tree_sitter import Language, Parser

# initialize the parser for the code
language_build_path = os.path.join(os.path.dirname(__file__), 'py-tree-sitter.so')
PY_LANGUAGE = Language(language_build_path, 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

input_file = "./annotated data/svamp/svamp_indexed.jsonl"
dev_file = "./annotated data/svamp/svamp_failed.jsonl"

def verify_code(code: str, gold_answer: str) -> bool:
    try:
        exec(code)
        if math.isclose(float(gold_answer), float(eval("answer"))):
            return True
        else:
            return False
    except Exception as e:
        return False

def get_code_from_answer_str(answer_str: str, equation_str: str) -> str:
    # reverse_var_dict only keeps the constants and the t_lines does not contain the constant inits
    reverse_var_dict: Dict[float, str] = {}
    reverse_temp_var_dict: Dict[float, str] = {}
    temp_var_num = 0
    t_lines = []

    def get_var_name(var_str: str, allow_new: bool = True) -> str:
        num = float(var_str)
        if num in reverse_temp_var_dict:
            var_name = reverse_temp_var_dict[num]
        elif num in reverse_var_dict:
            var_name = reverse_var_dict[num]
        elif allow_new:
            # a new constant
            var_name = f"n{len(reverse_var_dict)}"
            reverse_var_dict[num] = var_name
        else:
            raise ValueError(f"{var_str} not found in var/temp dict")

        return var_name

    def get_node_text(node, text) -> str:
        return text[node.start_byte: node.end_byte]

    expression = equation_str

    # interpret the formula with a parse tree
    assert expression.isascii, f"{expression} is not ascii"
    parsed_tree = parser.parse(bytes(expression, 'utf-8'))

    # do a dfs on the parsed tree to get the values replaced with names
    formula_bits = []
    node_stack = [parsed_tree.root_node.children[0].children[0]]
    while len(node_stack) > 0:
        node = node_stack.pop()

        if node.type in ["integer", "float"]:
            var_name = get_var_name(get_node_text(node, expression))
            formula_bits.append(var_name)
        elif node.type in ["+", "-", "*", "/", "**", "(", ")", "//"]:
            formula_bits.append(get_node_text(node, expression))
        elif node.type in ["binary_operator", "parenthesized_expression"]:
            node_stack.extend(node.children[::-1])
        elif node.type == "unary_operator":
            if node.children[0].type == "+":
                var_name = get_var_name(get_node_text(node, expression))
                formula_bits.append(var_name)
            elif node.children[0].type == "-":
                val = -float(get_node_text(node, expression))
                if val in reverse_temp_var_dict or val in reverse_var_dict:
                    formula_bits.append(get_var_name(val, allow_new=False))
                elif -val in reverse_temp_var_dict or val in reverse_var_dict:
                    formula_bits.append("-"+get_var_name(-val, allow_new=False))
                else:
                    formula_bits.append(get_var_name(val, allow_new=True))
            else:
                raise ValueError(f"{expression} has unary operator {node.children[0].type}")    
        else:
            raise ValueError(f"{expression} has {node.type}")

    right_formula = "".join(formula_bits)
        
    # add the temporary var
    # NOTE: we can't use the len(reverse_temp_var_dict) because we may have the same temp var in different lines
    temp_var_name = f"t{temp_var_num}"
    temp_var_num += 1
    reverse_temp_var_dict[float(answer_str)] = temp_var_name

    # add the line
    t_lines.append(f"{temp_var_name}={right_formula}")

    # add the const var inits
    init_lines = []
    sorted_var_dict = sorted(reverse_var_dict.items(), key=lambda x: int(x[1][1:]))
    for var_val, var_name in sorted_var_dict:
        # if the float var is not directly used, and it can be casted as int, do cast as init
        if not str(var_val) in equation_str and math.isclose(int(var_val), var_val, abs_tol=1e-4):
            init_lines.append(f"{var_name}={int(var_val)}")
        else:
            init_lines.append(f"{var_name}={var_val}")


    if len(t_lines) == 0:
        # no <<formula>> are given for this example, simply skip
        return "NULL"

    # replace the last line's temp var name with "answer"
    t_lines[-1] = "answer=" + t_lines[-1].split("=")[1]

    return "\n".join(init_lines + t_lines)

def verify_svamp(instances: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    # verify the validity of the code
    failed_code_execution_indices = []
    for instance in instances:
        if not verify_code(instance["code"], instance["answer"]):
            failed_code_execution_indices.append(instance)

    return failed_code_execution_indices

if __name__ == "__main__":
    # load the dev data
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
 
    processed_data = verify_svamp(data)

    # write processed data to file
    with open(dev_file, "w") as f:
        f.write("\n".join([json.dumps(data) for data in processed_data]))
        # f.write(str(dev_lines))