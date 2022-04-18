import re
import os
import json
import math
import logging

from typing import List, Dict, Any
from tqdm import tqdm

from execution.execution_evaluation import mathqa_execution, execution_acc
from tree_sitter import Language, Parser

# initialize the parser for the code
language_build_path = os.path.join(os.path.dirname(__file__), 'py-tree-sitter.so')
PY_LANGUAGE = Language(language_build_path, 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

#set file to evaluate
input_file = "./annotated data/gsmath/gsmath_failed_train.jsonl"
# input_file = "./annotated data/gsmath/test_errors.jsonl"

def get_answer_from_answer_str(answer_str: str) -> float:
    result_str = answer_str.split("\n")[-1].split(" ")[-1]
    result = float(result_str.replace(",", ""))

    return result

def get_code_from_answer_str(answer_str: str, question_str: str) -> str:
    # reverse_var_dict only keeps the constants and the t_lines does not contain the constant inits
    reverse_var_dict: Dict[float, str] = {}
    reverse_temp_var_dict: Dict[float, str] = {}
    temp_var_num = 0
    t_lines = []

    for line in answer_str.split("\n")[:-1]:
        if not ("<<" in line and ">>" in line):
            continue

        # first extract the formula
        formula = line[line.index("<<") + 2: line.index(">>")]

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

        # make sure that the formula is valid
        expression, result = formula.split("=")
        if "/" in result:
            result = eval(result)
        if not math.isclose(eval(expression), float(result)):
            logging.debug(f"expression: {eval(expression)} result: {float(result)}")
            return "NULL"

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
        reverse_temp_var_dict[float(result)] = temp_var_name

        # add the line
        t_lines.append(f"{temp_var_name}={right_formula}")

    # add the const var inits
    init_lines = []
    sorted_var_dict = sorted(reverse_var_dict.items(), key=lambda x: int(x[1][1:]))
    for var_val, var_name in sorted_var_dict:
        # if the float var is not directly used, and it can be casted as int, do cast as init
        if not str(var_val) in question_str and math.isclose(int(var_val), var_val, abs_tol=1e-4):
            init_lines.append(f"{var_name}={int(var_val)}")
        else:
            init_lines.append(f"{var_name}={var_val}")


    if len(t_lines) == 0:
        # no <<formula>> are given for this example, simply skip
        return "NULL"

    # replace the last line's temp var name with "answer"
    t_lines[-1] = "answer=" + t_lines[-1].split("=")[1]

    return "\n".join(init_lines + t_lines)

def verify_code(code: str, gold_answer: str) -> bool:
    try:
        exec(code)
        if math.isclose(float(gold_answer), float(eval("answer"))):
            return True
        else:
            return False
    except Exception as e:
        return False

def process_gsmath(instances: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    failed_code_extraction_indices = []

    # updates the code and answer in the logged error messages
    for i, instance in tqdm(enumerate(instances)):
        instance["code"] = get_code_from_answer_str(instance["original_answer"], instance["text"])
        instance["answer"] = get_answer_from_answer_str(instance["original_answer"])

    # verify the validity of the code
    failed_code_execution_indices = []
    for i, instance in enumerate(instances):
        if instance["code"] == "NULL":
            failed_code_extraction_indices.append(i)
            logging.debug(f"{instance['task_id']} failed to extract, " \
                  f"original_answer: {instance['original_answer']}, " \
                  f"code: \n{instance['code']}\nanswer: {instance['answer']}")

        elif not verify_code(instance["code"], instance["answer"]):
            failed_code_execution_indices.append(i)
            # logs the failed item
            logging.debug(f"{instance['task_id']} failed to verify, " \
                  f"original_answer: {instance['original_answer']}, " \
                  f"code: \n{instance['code']}\nanswer: {instance['answer']}")

    all_failed_indices = sorted(failed_code_extraction_indices + failed_code_execution_indices)

    print(f"{len(failed_code_extraction_indices)}/{len(instances)} failed to extract code")
    print(f"{len(failed_code_execution_indices)}/{len(instances)} failed to execute to the correct result")
    print(f"{len(all_failed_indices)}/{len(instances)} failed in total")


if __name__ == "__main__":
    # configure logging
    if (os.path.exists("failed.log")):
        os.remove("failed.log")
    logging.basicConfig(filename='failed.log',level=logging.DEBUG)

    # load the input file for comparison
    with open(input_file, "r") as f:
        input_lines = f.readlines()
        input_data = [json.loads(line) for line in input_lines]

    # process all the data
    # note that this doesn't change the failed file, so the code will need to be swapped for the correct code at some point
    process_gsmath(input_data)