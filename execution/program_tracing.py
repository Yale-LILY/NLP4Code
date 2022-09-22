import math
import scipy
import json
import os

from concurrent.futures import ProcessPoolExecutor as Pool
from typing import List, Dict, Tuple, Any, Union, NamedTuple, Set
from scipy import special

from typing import List, Dict, Any
from tqdm import tqdm
from finetuning.lightning_modules.datasets.reader_utils import get_statements_from_code, byte_idx_to_char_idx
from execution.safe_execution_util import execute, canonicalize_var_dict, simple_canonicalize_var_dict
from tree_sitter import Language, Parser

ProgState = Dict[str, float]
HashableProgState = Set[float]
ProgTraceUnit = NamedTuple("ProgTraceUnit", [("code", str), ("type", str), ("state", ProgState)])
ProgTrace = List[ProgTraceUnit]
Program = NamedTuple("Program", [("code", str), ("code_lite", str), ("trace", ProgTrace)])

# initialize the parser for the code
language_build_path = os.path.join(os.path.dirname(__file__)+'/../preprocessing/', 'py-tree-sitter.so')
PY_LANGUAGE = Language(language_build_path, 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

"""
Tracing the execution of a program:
    1. It parses the program into a sequence of tracing units (currently stmts);
    2. Make some markings of the tracing units;
    3. Insert tracing code to the program, after every tracing unit;
    4. Run the program with tracing;
    5. Collect the variable tracing information.
"""

from copy import deepcopy
from types import ModuleType

tracing_local_list = []
def record_state(local_var_dict):
    copied_local_var_dict = simple_canonicalize_var_dict(local_var_dict)
    tracing_local_list.append(copied_local_var_dict)

def get_function_final_state(program: str) -> Dict[str, Any]:
    # first parse the program with tree-sitter
    stmts = get_statements_from_code(program, parser)

    if stmts is None:
        return {"result": "ERROR: unparseable"}
    
    # put a record state before every return point
    program_with_tracing = ""
    program_bytes = bytes(program, "utf-8")
    byte_idx = 0
    for stmt in stmts:
        stmt_str = program_bytes[byte_idx:stmt['end_byte']+1].decode("utf-8")
        if stmt["type"] == "return_statement":
            # build the harness code
            return_token_idx = stmt_str.find("return")
            return_val_expr = stmt_str[return_token_idx:].replace("return", "").strip().strip(";")
            
            if len(return_val_expr) > 0:
                harness_code = f"_return_val={return_val_expr}; record_state(locals()); "

                # insert into the original return stmt
                # stmt_str = stmt_str[:return_token_idx] + harness_code + \
                #             stmt_str[return_token_idx:].replace(return_val_expr, " _return_val")
                stmt_str = stmt_str[:return_token_idx] + harness_code + "return _return_val\n"
            else:
                harness_code = f"record_state(locals()); "
                stmt_str = stmt_str[:return_token_idx] + harness_code + stmt_str[return_token_idx:]
        
        program_with_tracing += stmt_str
        byte_idx = stmt['end_byte']+1


    # execute the program with tracing code
    tracing_result = execute(program_with_tracing, {}, globals={
                              "tracing_local_list": tracing_local_list,
                              "record_state": record_state,
                              }, use_tracing=True, timeout=10, output_locals=False)
    
    return tracing_result

def assertion_to_test(assertion: str) -> str:
    """ get rid of the expected results in the assertion """
    program_bytes = bytes(assertion, 'utf-8')
    parsed_tree = parser.parse(program_bytes)

    root_node = parsed_tree.root_node
    assert len(root_node.children) == 1

    assert_stmt = root_node.children[0]
    assert assert_stmt.type == "assert_statement"
    # assert len(assert_stmt.children) == 2 # NOTE: it might break if something like "assert a == b,c"

    comparison_stmt = assert_stmt.children[1]
    assert comparison_stmt.type == "comparison_operator"
    assert len(comparison_stmt.children) == 3

    call_stmt = comparison_stmt.children[0]
    while call_stmt.type == "parenthesized_expression":
        assert len(call_stmt.children) == 3
        call_stmt = call_stmt.children[1]
    assert call_stmt.type == "call"

    call_str = program_bytes[call_stmt.start_byte:call_stmt.end_byte].decode("utf-8").strip()

    return call_str



def get_execution_states(program: str, debug=False) -> Union[ProgTrace, None]:
    # first parse the program with tree-sitter
    stmts = get_statements_from_code(program, parser)

    if stmts is None:
        if debug:
            print(f'skipping unparseable example')
            print(f"##########\n{program}\n##########")
        return None

    # extract the stmt strings
    idx = 0
    stmt_states = []
    for stmt in stmts:
        start_idx = byte_idx_to_char_idx(stmt['start_byte'], program)
        end_idx = byte_idx_to_char_idx(stmt['end_byte'], program)

        if start_idx != idx:
            # add the gap
            stmt_states.append({"code": program[idx:start_idx], "type": "gap"})

        # add the stmt
        stmt_states.append({"code": program[start_idx:end_idx], "type": "stmt"})
        idx = end_idx


    # NOTE: FIXME: This only works for straight-line programs since it does not consider indentation
    for stmt in stmt_states:
        if stmt["type"] == "stmt":
            stmt["code"] += "\nrecord_state(locals())"

    # now assemble the program back together
    traced_program = "".join([stmt["code"] for stmt in stmt_states])

    # execute the program with tracing code
    result = execute(traced_program, {}, 
                     globals={"tracing_local_list": tracing_local_list,
                              "record_state": record_state}, use_tracing=True)

    if result["result"] == "passed":
        # add the *output* states for each statement and remove the tracing code to restore orginal program
        stmt_idx = 0
        for stmt in stmt_states:
            if stmt["type"] == "stmt":
                stmt["execution_state"] = result["tracing_local_list"][stmt_idx]
                stmt["code"] = stmt["code"].replace("\nrecord_state(locals())", "")
                stmt_idx += 1
        prog_trace = [ProgTraceUnit(stmt["code"], stmt["type"], 
                        stmt["execution_state"] if stmt["type"] == "stmt" else {}) for stmt in stmt_states]
        return prog_trace
    else:
        if debug:
            print(f'skipping example of error: {result["result"]}')
            print(f"##########\n{program}\n##########")
        return None

def batch_program_tracing(programs: List[str], n_processes=20) -> List[Union[ProgTrace, None]]:
    with Pool(n_processes) as p:
        tracing_outputs = p.map(get_execution_states, programs)
    return list(tracing_outputs)

def exec_stmt_in_context(stmt: str, context: Dict[str, Any]):
    # NOTE: FIXME: This only works for straight-line programs since it does not consider indentation
    traced_stmt = stmt + "\nrecord_state(locals())"

    # execute the program with tracing code
    if "math" in context:
        context["math"] = math
    if "scipy" in context:
        context["scipy"] = scipy
        context["scipy.special"] = special

    result = execute(traced_stmt, locals=context, 
                     globals={"tracing_local_list": tracing_local_list, "deepcopy": deepcopy, 
                              "record_state": record_state, "ModuleType": ModuleType}, use_tracing=True)

    if result["result"] == "passed":
        # return the execution states as the local var list
        assert len(result["tracing_local_list"]) == 1, f"tracing_local_list: {result['tracing_local_list']}"
        stmt_output_state = result["tracing_local_list"][0]
        return stmt_output_state
    else:
        return None

def is_trivial_state(state_dict: Dict[str, Any], prev_stmt: str):
    if len(state_dict) == 0:
        return True

    assert prev_stmt is not None, "prev_stmt must be provided to determine trivial states unless the state is empty"

    if prev_stmt.split(" ")[0] in ["#", "import"]:
        return True

    assert len(state_dict) == 1, f"prev_stmt {prev_stmt}; original state dict {state_dict}"

    return f"{list(state_dict.keys())[0]} = {list(state_dict.values())[0]}" in prev_stmt

def get_state_repr(state_dict: Dict[str, Any], prev_stmt: str = None, only_include_keys: List[str] = None, 
                   prev_state_dict: Dict[str, Any] = None, use_diff=False, skip_trivial_states: bool = False):
    if use_diff:
        raise NotImplementedError

    if only_include_keys is not None:
        state_dict = {k: v for k, v in state_dict.items() if k in only_include_keys}

    if skip_trivial_states and is_trivial_state(state_dict, prev_stmt):
        return ""

    repr = "# "
    for key, value in state_dict.items():
        repr += f"{key} = {value}; "
    repr += "\n"

    return repr

def test1():
    # load some sample programs
    with open('data/mathqa/val-python.jsonl', 'r') as f:
        lines = f.readlines()

        json_examples = [json.loads(line) for line in lines]

    with open('data/mathqa/val_python_with_states.jsonl', 'w+') as f:
        success_count = 0
        failed_count = 0
        for json_example in tqdm(json_examples):

            program = json_example["code"]
            stmt_states = get_execution_states(program)

            if stmt_states is not None:
                json_example["states"] = stmt_states
                f.write(json.dumps(json_example) + "\n")
                success_count += 1
            else:
                failed_count += 1

        print(f"Successfully traced {success_count}/{success_count+failed_count} programs")

def test2():
    with open("results/mbpp-codex_davinci-few_shot-test-pass_at_100-1_test/predictions_step_0_rank_0.jsonl", "r") as f:
        dev_data = [json.loads(line) for line in f.readlines()]
    
    success = 0
    total = 0
    
    # failed_indices = [289, 322, 197, 70, 326, 42, 300, 80, 242, 118, 186, 155]
    # failed_indices = [242, 326]
    # dev_data = [dev_data[i] for i in failed_indices]

    # failed_indices = set()

    for example in tqdm(dev_data):
        generated_programs = [x['program'].strip() for x in example["generated_k_programs"]]
        tests = example["metadata"]["test_list"]

        for t in tests:
            for sol in generated_programs:
                program = sol + "\n\n" + example["metadata"]["test_setup_code"] + "\n\n" + assertion_to_test(t)
                # tracing_result = get_function_final_state(program.strip())

                # total += 1
                # if tracing_result["result"] == "passed":
                #     success += 1
                # else:
                    # print(f"failed to trace program: {tracing_result['result']}")
                    # failed_indices.add((total-1) // 3)
                    # tracing_result = get_function_final_state(program.strip())
                    # print(f"##########\n{program}\n##########")
                    # break

        
                # print(f"Success rate: {success}/{total} = {success/total}")
    
    # print(f"failing indices: {failed_indices}")

def test3():
    with open("data/mbpp/mbpp_train.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    
    total, success = 0, 0
    for example in data:
        tests = example["test_list"]

        for t in tests:
            program = example["code"] + "\n\n" + example["test_setup_code"] + "\n\n" + assertion_to_test(t)
            exec_result = execute(program, timeout=10, output_locals=False)
            total += 1

            if exec_result["result"] == "passed":
                success += 1
            else:
                print(f"failed to execute program: {exec_result['result']}")
                print(f"the program is: ##########\n{program}\n##########")

            print(f"Success rate: {success}/{total} = {success/total}")


if __name__ == "__main__":
    test2()