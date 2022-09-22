import os
import json
import subprocess
import sqlite3
import shutil
import re
from time import time
from pathlib import Path

import pandas as pd

from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from multiprocessing import Process, Manager

from tree_sitter import Language, Parser
from func_timeout import func_timeout, FunctionTimedOut

from execution.spider_execution import spider_answer_eq, spider_official_execution_sql, step_wise_execution_sql, squall_answer_eq, flatten_list_of_list
from execution.spider_execution import spider_official_execution_sql, spider_execution_pd_sql, step_wise_thread_execution_sql, squall_execution_sql, squall_official_execution_sql
from execution.spider_official_exec_match import eval_exec_match
from execution.wtq_eval import is_number, wtq_answer_eq


# DB_DIR = "data/spider/database"
# DB_DIR = "data/squall/tables/db"

# initialize the parser for the code
language_build_path = os.path.join(os.path.dirname(__file__), 'sqlite-tree-sitter.so')
SQL_LANG = Language(language_build_path, 'sqlite')
parser = Parser()
parser.set_language(SQL_LANG)

def dump_db_schema():
    db_folder = 'data/spider/database'
    db_names = [f.name for f in os.scandir(db_folder) if f.is_dir()]

    for db_name in tqdm(db_names):
        db_file = f'{db_folder}/{db_name}/{db_name}.sqlite'
        dump_file = f'{db_folder}/{db_name}/{db_name}_dump.sql'
        if os.path.exists(dump_file):
            raise Exception(f'{dump_file} already exists')
            # dump the database
        command = f'sqlite3 {db_file} .dump > {dump_file}'
        # subprocess.run([command], stdout=subprocess.DEVNULL)
        os.system(command)

def cleanup():
    db_folder = 'data/spider/database'
    db_names = [f.name for f in os.scandir(db_folder) if f.is_dir()]

    check = 0
    for db_name in tqdm(db_names):
        dump_file = f'{db_folder}/{db_name}/{db_name}_dump.sql'
        if os.path.exists(dump_file):
            # dump the database
            # command = f'rm -f {dump_file}'
            # subprocess.run(command.split(' '), stdout=subprocess.DEVNULL)
            check += 1
    
    print(f'{check} files exist')

def get_mini_dev():
    # get the 200 dev examples that are used in previous experiments
    with open('results_cot/spider_codex_cot_results_sql_8_crafted_shot-200_baseline_results.jsonl', 'r') as f:
        results = [json.loads(s) for s in f.readlines()]

    # get the mini dev examples out of it
    mini_dev = [result['example'] for result in results]

    with open('data/spider/dev_mini_processed.jsonl', 'w+') as f:
        for ex in mini_dev:
            f.write(json.dumps(ex) + '\n')

    print(f"{len(mini_dev)} mini dev examples dumped")

def get_few_shot_examples():
    few_shot_questions = [
        "How many distinct kinds of injuries happened after season 2010?",
        "Return the hosts of competitions for which the theme is not Aliens?",
        "Show the average, maximum, minimum enrollment of all schools.",
        "Return the ids and details corresponding to projects for which there are more than two documents.",
        "List the type of the services in alphabetical order.",
        "Show the average price range of hotels that have 5 star ratings and allow pets.",
        "Find all the phone numbers.",
        "Which transportation method is used the most often to get to tourist attractions?",
    ]

    with open('data/spider/train_spider_processed_v2.jsonl', 'r') as f:
        all_train_data = [json.loads(s) for s in f.readlines()]
    
    few_shot_examples = list(filter(lambda x: x['question'] in few_shot_questions, all_train_data))

    with open('data/spider/few_shot_examples.jsonl', 'w+') as f:
        for ex in few_shot_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"{len(few_shot_examples)} few shot examples found and dumped")

compound_keywords = ["INTERSECT", "UNION", "EXCEPT"]

def node_str(node, str_bytes):
    return str_bytes[node.start_byte:node.end_byte].decode('utf-8')

def get_identifiers(node, identifiers_nodes: list):
    if node.type == 'identifier' or node.type == '.':
        identifiers_nodes.append(node)
        return
    elif node.type == 'function_name':
        return
    elif node.type == 'from_clause':
        return
    elif node.type == 'table_or_subquery':
        assert node.children[0].type == 'identifier'
        return
    else:
        for child in node.children:
            get_identifiers(child, identifiers_nodes)

def prune_identifiers(node, str_bytes):
    # first get all the identifier nodes
    identifier_nodes = []
    get_identifiers(node, identifier_nodes)

    # then do some aggregation for the dot stuff
    identifier_strs = []
    for node in identifier_nodes:
        node_s = node_str(node, str_bytes)
        if len(identifier_strs) == 0:
            identifier_strs.append(node_s)
        elif node.type == '.' or identifier_strs[-1].endswith('.'):
            identifier_strs[-1] += node_s
        else:
            identifier_strs.append(node_s)
    
    return identifier_strs

def remove_alias(node, sql_bytes, str_list: List[str]):
    if len(node.children) == 0:
        s = node_str(node, sql_bytes)
        if len(str_list) > 0 and str_list[-1] == '.':
            str_list.pop(-1)
            str_list.pop(-1)
            str_list.append(s)
        else:
            str_list.append(s)
    else:
        for child in node.children:
            if 'literal' in child.type:
                str_list.append(node_str(child, sql_bytes))
            else:
                remove_alias(child, sql_bytes, str_list)

def recursive_decompose(node, sql_bytes, result_stmts: List[str], tmp_tab_n) -> int:
    # returns the tmp table identifier id
    
    # in case of error
    if node.type == 'ERROR':
        return -1

    # ignore the first few layers
    if node.type in ['sql_stmt_list', 'sql_stmt']:
        assert len(node.children) == 1 or node.children[1].type == ';' or any([n.type == 'ERROR' for n in node.children])
        return recursive_decompose(node.children[0], sql_bytes, result_stmts, tmp_tab_n)

    # if this is a compound statement, we need to decompose it
    if any([child.type in compound_keywords for child in node.children]):
        idx, connect_node = list(filter(lambda x: x[1].type in compound_keywords, enumerate(node.children)))[0]

        left_bytes = sql_bytes[node.start_byte:node.children[idx-1].end_byte]
        right_bytes = sql_bytes[node.children[idx+1].start_byte:node.end_byte]

        left_tmp_tab = recursive_decompose(parser.parse(left_bytes).root_node, left_bytes, result_stmts, tmp_tab_n)
        right_tmp_tab = recursive_decompose(parser.parse(right_bytes).root_node, right_bytes, result_stmts, left_tmp_tab)
        
        # add the compound stmt at last
        final_tmp_tab = right_tmp_tab + 1
        result_stmts.append(f'CREATE TABLE TMP_TAB_{final_tmp_tab} AS ' + \
            f'SELECT * FROM TMP_TAB_{left_tmp_tab} {connect_node.type} SELECT * FROM TMP_TAB_{right_tmp_tab}')
        return final_tmp_tab

    # by this point, it should all be simple queries with no compound factor
    from_nodes = list(filter(lambda x: x.type == 'from_clause', node.children))
    if len(from_nodes) == 0:
        return -1
    from_node = from_nodes[0]

    # from clause may have JOIN that needs to be decomposed
    if any([child.type == 'join_operator' for child in from_node.children]):
        # replace all the aliases with real table names
        # ts_query = SQL_LANG.query("(table_or_subquery (identifier) @tab-name (AS) (identifier) @tab-alias)")
        # ts_captures = ts_query.captures(from_node)
        # alias_dict = {}
        # for i in range(0, len(ts_captures), 2):
        #     assert ts_captures[i][1] == 'tab-name'
        #     assert ts_captures[i+1][1] == 'tab-alias'
        #     alias_dict[ts_captures[i+1][0]] = ts_captures[i][0]

        # gather all the information needed later by getting all the identifies
        identifier_strs = set(prune_identifiers(node, sql_bytes))

        # emit the JOIN operation sub select
        tmp_tab_n += 1
        join_op = f'CREATE TABLE TMP_TAB_{tmp_tab_n} AS SELECT {", ".join(identifier_strs)} {node_str(from_node, sql_bytes)}'
        result_stmts.append(join_op)

        # substitute the join operator with the tmp table identifier
        subbed_select = node_str(node, sql_bytes).replace(node_str(from_node, sql_bytes), f'FROM TMP_TAB_{tmp_tab_n}')
        # remove all the alias and dot stuff
        new_sql_bytes = bytes(subbed_select, 'utf-8')
        new_seq_str_list = []
        remove_alias(parser.parse(new_sql_bytes).root_node, new_sql_bytes, new_seq_str_list)
        subbed_select = ' '.join(new_seq_str_list)
        new_sql_bytes = bytes(subbed_select, 'utf-8')
        return recursive_decompose(parser.parse(new_sql_bytes).root_node, new_sql_bytes, result_stmts, tmp_tab_n)

    # by this point, it should have no join
    where_nodes = list(filter(lambda x: x.type == 'where_clause', node.children))
    if len(where_nodes) > 0:
        assert len(where_nodes) == 1
        where_node = where_nodes[0]

        # where clause may have nested select that needs to be decomposed
        if any([child.type == 'select_stmt' for child in where_node.children]):
            # first recursively decompose the nested select (since there might be more nested ones)
            nested_select_node = list(filter(lambda x: x.type == 'select_stmt', where_node.children))[0]
            tmp_tab_n = recursive_decompose(nested_select_node, sql_bytes, result_stmts, tmp_tab_n)

            # then emit the parent select
            subbed_select = node_str(node, sql_bytes).replace(node_str(nested_select_node, sql_bytes), 
                                                                f'SELECT * FROM TMP_TAB_{tmp_tab_n}')
            tmp_tab_n += 1
            result_stmts.append(f'CREATE TABLE TMP_TAB_{tmp_tab_n} AS ' + subbed_select)
            return tmp_tab_n
    
    # by this point, it should have no nested select
    tmp_tab_n += 1
    result_stmts.append(f'CREATE TABLE TMP_TAB_{tmp_tab_n} AS ' + node_str(node, sql_bytes))
    return tmp_tab_n


def decompose_sql(sql: str, expect_error: bool = False) -> List[str]:
    ori_sql_bytes = bytes(sql, 'utf-8')
    parsed_tree = parser.parse(ori_sql_bytes)

    # do a dfs on the parsed tree to record all the simple statements
    result_stmts: List[str] = []
    return_tmp_tab_num = recursive_decompose(parsed_tree.root_node, ori_sql_bytes, result_stmts, 0)

    if return_tmp_tab_num == -1:
        if expect_error:
            return []
        else:
            raise Exception(f'Error in decomposing sql: {sql}')

    # prune out the same table
    stmt_var_dict = {}
    var_sub_dict = {}
    for stmt in result_stmts:
        tmp_tab_var = stmt.split(' AS ')[0].split(' ')[-1]
        select_stmt = ' AS '.join(stmt.split(' AS ')[1:])

        if select_stmt in stmt_var_dict:
            # build var sub dict, e.g., {'TMP_TAB_3': 'TMP_TAB_1'}
            var_sub_dict[tmp_tab_var] = stmt_var_dict[select_stmt]
        else:
            # sub the vars
            for old_var, new_var in var_sub_dict.items():
                select_stmt = select_stmt.replace(old_var, new_var)
            stmt_var_dict[select_stmt] = tmp_tab_var

    # reverse the dict and order it to recover the program
    var_stmt_dict = {v: k for k, v in stmt_var_dict.items()}
    new_stmts = []
    for var, stmt in var_stmt_dict.items():
        new_stmts.append(f"CREATE TABLE {var} AS {stmt}")

    return new_stmts

def tab_schema_only(df: pd.DataFrame) -> str:
    return ', '.join(df.columns)

def test_sql_decomposition():
    with open("data/spider/train_spider_processed_v2.jsonl", "r") as f:
        examples = [json.loads(s) for s in f.readlines()]

    not_match = 0
    exec_err = 0
    no_decomp = 0
    for example in tqdm(examples):
        sql = example['query'].replace("\"", "'")

        # decomp_sqls = decompose_sql(sql)
        
        # # sequentially execute the sqls
        # db_id = example["db_id"]
        # db_path = os.path.join(DB_DIR, db_id, db_id + ".sqlite")

        # # copy file to be safe
        # tmp_db_path = os.path.join(DB_DIR, 'tmp', 'tmp' + ".sqlite")
        # shutil.copyfile(db_path,tmp_db_path) 
        # conn = sqlite3.connect(f"file:{tmp_db_path}?mode=rw", uri=True)
        # cursor = conn.cursor()

        # for decomp_sql in decomp_sqls:
        #     try:
        #         cursor.execute(decomp_sql)
        #         error_msg = None
        #     except sqlite3.OperationalError as e:
        #         error_msg = f"ERROR: {str(e)}"
        #         # print(f"Error {str(e)} in execution sql query {sql}")

        decomp_sqls = get_sql_exec_result_decomp(sql, example)

        if decomp_sqls is None:
            exec_err += 1
            continue 

        last_tmp_tab_var = decomp_sqls[-2].split(' AS ')[0].split(' ')[-1]
        
        # last_tmp_tab_var = 'TMP_TAB_' + decomp_sqls[-1].split('TMP_TAB_')[1].split(' ')[0]
        example["db_id"] = 'tmp'
        match = spider_official_execution_sql(f"SELECT * FROM {last_tmp_tab_var}", example, keep_distinct=True)

        # exec_result = pd.read_sql_query(f"SELECT * FROM {last_tmp_table_number}", conn)

        if len(decomp_sqls) > 6:
            print('-'*40)
            print(f"original_sql: \n{sql}")
            print('-'*40)
            print(f"decomposed: \n")
            for decomp_sql in decomp_sqls:
                print(decomp_sql)
            print('-'*40)

        if len(decomp_sqls) == 2:
            no_decomp += 1
        
        if not match:
            not_match += 1

        # conn.close()
    
    print(f"not match: {not_match}/{len(examples)}")
    print(f"exec err: {exec_err}/{len(examples)}")
    print(f"no decomp: {no_decomp}/{len(examples)}")

def pd_df_to_dict(df: pd.DataFrame) -> Tuple[dict, bool]:
    cutoff = False
    if len(df) > 5:
        df = df.head(5)
        cutoff = True
    return df.to_dict(orient='tight'), cutoff

def pd_df_from_dict(dt: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(dt, orient='tight')

def get_sql_exec_result_decomp(code: str, example: Dict[str, Any], use_thread: bool = False):
    decomposed_sql_stmts = code.split("\n")

    # copy file to be safe
    db_path = example["db_path"]
    extension = db_path.split(".")[-1]
    db_path_root = Path(db_path).parent.parent.absolute()

    random_file_name = int(abs(hash(db_path+example["query"]+example["question"]))) % 10000000
    tmp_db_path = os.path.join(str(db_path_root), 'tmp', f"tmp_{random_file_name}.{extension}")
    shutil.copyfile(db_path, tmp_db_path) 
    conn = sqlite3.connect(f"file:{tmp_db_path}?mode=rw", uri=True)

    # sequentially execute the sqls
    exec_states = []
    exec_cutoff_bools = []
    error_occured = False
    for stmt in decomposed_sql_stmts:
        if not use_thread:
            state, error_msg = step_wise_execution_sql(stmt, conn)
        else:
            state, error_msg = step_wise_thread_execution_sql(stmt, conn)
        if error_msg is not None: 
            execution_result_store = "ERROR"
            execution_result_cutoff = False
            error_occured = True
        else:
            execution_result_store, execution_result_cutoff = pd_df_to_dict(state)
        
        exec_states.append(execution_result_store)
        exec_cutoff_bools.append(execution_result_cutoff)

    # cleanup
    conn.commit()
    conn.close()
    
    tmp_tab_var = decomposed_sql_stmts[-1].split(' AS ')[0].split(' ')[-1]
    final_sql = f"SELECT * FROM {tmp_tab_var}"

    if not error_occured:
        execution_match = bool(eval_exec_match(tmp_db_path, db_path, final_sql, metadata["query"], plug_value=False, keep_distinct=False, progress_bar_for_each_datapoint=False))
    else:
        execution_match = False

    os.remove(tmp_db_path)

    return exec_states, exec_cutoff_bools, execution_match

def get_sql_exec_result(code: str, metadata):
    # execute the code to append the result
    execution_result = spider_execution_pd_sql(code, metadata)
    if execution_result is None:
        execution_result_store = "ERROR"
        execution_result_cutoff = False
        execution_match = False
    else:
        execution_result_store, execution_result_cutoff  = pd_df_to_dict(execution_result)
        execution_match = bool(squall_official_execution_sql(code, metadata))

    return execution_result_store, execution_result_cutoff, execution_match

    
def process_spider_output_example(code: str, metadata: dict, 
                                  distinct_programs_counts: dict = None,
                                  use_decomp_sql: bool = False) -> dict:
    # process the raw code outputs
    code = code.split("\n\n")[0].strip()
    code = code.replace("\"", "'")
    lower_code = re.sub(r"\b(?<!')(\w+)(?!')\b", lambda match: match.group(1).lower(), code) 

    # do not emit the same program
    if distinct_programs_counts is not None:
        if lower_code in distinct_programs_counts:
            distinct_programs_counts[lower_code] += 1
            return None
        else:
            distinct_programs_counts[lower_code] = 1

    if use_decomp_sql:
        execution_result_store, execution_result_cutoff, execution_match = get_sql_exec_result_decomp(code, metadata, use_thread=True)
    else:
        try:
            execution_result_store, execution_result_cutoff, execution_match = func_timeout(10, get_sql_exec_result, args=(code, metadata))
        except FunctionTimedOut:
            execution_result_store = "ERROR"
            execution_result_cutoff = False
            execution_match = False
    
    # construct the program dictionary
    program_dict = {"code": code, "lower_code": lower_code, 
                    "exec_result": execution_result_store, 
                    "exec_result_cutoff": execution_result_cutoff,
                    "exec_match": execution_match}
    
    return program_dict

def postprocess_spider_output(input_file_paths: List[str], output_file_path: str, use_decomp_sql: bool = False):
    examples = []
    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as f:
            examples.extend([json.loads(s) for s in f.readlines()])
    
    # # automatic recover the postprocessing
    # with open(output_file_path, "r") as f:
    #     processed_examples = []
    #     for s in f.readlines():
    #         try:
    #             json_obj = json.loads(s) 
    #             processed_examples.append(json_obj)
    #         except Exception as e:
    #             print(s)
    #             continue

    #     if len(processed_examples) > 0:
    #         processed_questions = [s["metadata"]["question"] for s in processed_examples]
            
    #         remaining_examples = list(filter(lambda s: s["question"] not in processed_questions, examples))
    #         print(f"recovered progress for {len(processed_examples)} examples, remaining examples: {len(remaining_examples)}")
            # examples = remaining_examples

    f = open(output_file_path, "w+")
    
    for output in tqdm(examples):
        metadata = output["example"]
        metadata["db_path"] = os.path.join("data/squall/tables/db", f'{metadata["db_id"]}.db')
        result_example_dict = {"metadata": metadata}

        processed_programs = []
        distinct_programs_counts = {}

        # store the information for the gold program
        gold_program = metadata["query"] if not use_decomp_sql else "\n".join(decompose_sql(metadata["query"]))
        result_example_dict["gold_program"] = process_spider_output_example(gold_program, metadata, use_decomp_sql=use_decomp_sql)
        assert result_example_dict["gold_program"] is not None

        # add all the generated results
        # NOTE: the generated programs may be the same as the gold one, but we keep them separated during processing
        for code in output["all_output"][0]: 
            # process the raw code outputs
            processed_result = process_spider_output_example(code, metadata, distinct_programs_counts, use_decomp_sql=use_decomp_sql)

            if processed_result is not None:
                processed_programs.append(processed_result)
            else:
                continue
        
        # add the program count
        for processed_program in processed_programs:
            processed_program["program_count"] = distinct_programs_counts[processed_program["lower_code"]]
        
        result_example_dict["generated_programs"] = processed_programs

        # write to file
        f.write(json.dumps(result_example_dict) + "\n")
    
    f.close()

def add_db_path(file_path: str, output_file_path: str):
    with open(file_path, "r") as f:
        examples = [json.loads(s) for s in f.readlines()]

    for example in examples:
        example["db_path"] = os.path.join("data/spider/database", example["db_id"], f'{example["db_id"]}.sqlite')
    
    with open(output_file_path, "w+") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

def pass_at_k_transform(file_path: str, output_file_path: str, k: int):
    with open(file_path, "r") as f:
        examples = [json.loads(s) for s in f.readlines()]

    for example in examples:
        print("")
    
    with open(output_file_path, "w+") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

def fix_evaluation_for_verification(file_path: str, output_file_path: str):
    with open(file_path, "r") as f:
        examples = [json.loads(s) for s in f.readlines()]

    fixed_example_n = 0
    for example in examples:

        for program_dict in example["generated_programs"]:
            if isinstance(program_dict["exec_result"], str):
                continue
            if program_dict["exec_match"]:
                continue
            exec_result = pd_df_from_dict(program_dict["exec_result"])
            list_exec_result = exec_result.values.tolist()
            list_exec_result = post_process_exec_result(list_exec_result, example["metadata"])
            if squall_answer_eq(list_exec_result, example["metadata"]["original_answer"]):
                program_dict["exec_match"] = True
                fixed_example_n += 1
    
    print(f"fixed {fixed_example_n} examples out of {len(examples)}, ratio is {fixed_example_n / len(examples)}")

    with open(output_file_path, "w+") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

def post_process_exec_result(exec_result: Any, metadata: Dict[str, Any]) -> Any:
    if len(exec_result) == 0 or exec_result is None or exec_result[0] is None or exec_result[0][0] is None:
        return exec_result

    if exec_result[0][0] in [0, 1]:
        if "above or below" in metadata["question"] or "below or above" in metadata["question"]:
            return [[["below", "above"][int(exec_result[0][0])]]]
        elif "more or less" in metadata["question"]:
            return [[["less", "more"][int(exec_result[0][0])]]]
        elif "before or after" in metadata["question"]:
            return [[["after", "before"][int(exec_result[0][0])]]]
        elif metadata["question"].split(" ")[0] in ["is", "are", "does", "do", "was", "were"]:
            return [[["no", "yes"][int(exec_result[0][0])]]]
        else:
            return exec_result
    elif "which month" in metadata["question"] and exec_result[0][0] in list(range(1, 13)):
        return [[["January", "February", "March", "April", "May", "June", "July", \
                  "August", "September", "October", "November", "December"][int(exec_result[0][0]) - 1]]]
    # elif is_number(exec_result[0][0]) and float(exec_result[0][0]) > 3100 and float(exec_result[0][0]).is_integer():
    #     return [[format(int(exec_result[0][0]), ",")]]
    else:
        return exec_result


def test_wtq_postprocessing(file_path):
    with open(file_path, "r") as f:
        examples = [json.loads(s) for s in f.readlines()]
    
    # examples = list(filter(lambda x: "month" in x["metadata"]["question"], examples))
    # examples = list(filter(lambda x: "less or more" in x["metadata"]["question"], examples))
    # examples = list(filter(lambda x: x["metadata"]["question"].startswith("is") or \
    #                                  x["metadata"]["question"].startswith("are") or \
    #                                  x["metadata"]["question"].startswith("does") or \
    #                                  x["metadata"]["question"].startswith("do"), examples))
    # examples = list(filter(lambda x: is_number(x["metadata"]["original_answer"]) and float(x["metadata"]["original_answer"]) >= 1000, examples))
    # examples = list(filter(lambda x: "%" in x["metadata"]["original_answer"], examples))

    match_count = 0
    for example in examples: 
        gold_program_dict = example["gold_program"]

        if isinstance(gold_program_dict["exec_result"], str):
            continue

        exec_result = pd_df_from_dict(gold_program_dict["exec_result"])
        list_exec_result = exec_result.values.tolist()
        list_exec_result = post_process_exec_result(gold_program_dict["code"], list_exec_result, example["metadata"])
        if wtq_answer_eq([flatten_list_of_list(list_exec_result)], example["metadata"]["original_answer"]):
            match_count += 1
        else:
            print('-'*20)
            print(f"question is {example['metadata']['question']}")
            print(f"sql is {gold_program_dict['code']}")
            print(f"predicted answer: {list_exec_result} but gold answer: {example['metadata']['original_answer']}")
            print('-'*20)
    
    print(f"{match_count} out of {len(examples)} examples, {match_count / len(examples)} percent matched")
        

if __name__ == "__main__":
    # main()
    # cleanup()
    # get_mini_dev()
    # get_few_shot_examples()

    # test_sql_decomposition()

    # input_files = [f"results/squall-t5_base-finetuning-cat_eval-pass_100_eval-max_pass_at_k-train_eval/predictions_step_0_rank_{i}.jsonl" for i in range(8)]
    # input_files = [f"results/squall-t5_base-finetuning-cat_eval-pass_20_eval-max_pass_at_k/predictions_step_1335_rank_{i}.jsonl" for i in range(8)]
    postprocess_spider_output(["results_cot/squall_codex_davinci-few_shot_baseline-for-in_house_models-pass@50-train.jsonl"], 
                              "data/squall/codex_50_output_train_verification.jsonl", use_decomp_sql=False)

    # add_db_path("data/spider/dev_processed.jsonl", "data/spider/dev_processed_db_path.jsonl")

    # fix_evaluation_for_verification("data/squall/t5_large_100_output_dev_verification.jsonl", "data/squall/t5_large_100_output_dev_verification_imp_eval.jsonl")

    # test_wtq_postprocessing("data/squall/t5_large_100_output_dev_verification.jsonl")
