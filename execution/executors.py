import os
import time
import ast
import sqlite3

from overrides import overrides
from typing import List, Any, Tuple, Dict, Set, Union
from concurrent.futures import ProcessPoolExecutor as Pool
from execution.program_tracing import assertion_to_test

from execution.spider_official_exec_match import eval_exec_match
from execution.spider_execution import spider_execution_pd_sql, pd_df_to_dict, spider_execution_py, db_to_df_dict, spider_answer_eq
from execution.safe_execution_util import execute
from execution.program_tracing import get_function_final_state
from execution.wtq_eval import wtq_execution_sql, wtq_answer_eq

"""
From the models' perspective, the model would only want two things: 
    1) if the execution result is right; 
    2) a way to output the execution result
"""

def sql_program_len(code: str) -> int:
    """ return the length of the sql query """
    return len(list(filter(lambda x: not len(x.strip()) == 0, code.split())))

def python_program_len(code: str) -> int:
    """ return the length of the python program """
    return len(list(filter(lambda x: not x.startswith("#") and not len(x.strip()) == 0, code.split("\n"))))

class BaseExecutor:
    def __init__(self, 
                 use_safe_exec: bool = False, 
                 use_cache: bool = True, 
                 n_processes: int = 20):
        self.use_safe_exec = use_safe_exec
        self.use_cache = use_cache
        self.n_processes = n_processes

        if self.use_cache:
            self.cache = {}
        
        self.stats_dict = {}
    
    def cache_key_func(self, program: str, example: Dict[str, Any]) -> str:
        raise NotImplementedError("BaseExecutor is an abstract class")

    @staticmethod
    def real_exec_program(program: str, example: Dict[str, Any]) -> Tuple[int, Union[str, List, Dict]]:
        """
        We need to use a pool to execute the programs, so we need to use static method and not a member method
        This is the real execution function that should be implemented by subclasses
        """
        raise NotImplementedError("BaseExecutor is an abstract class")
    
    def program_len(self, program: str) -> int:
        return len(program)

    def gold_program_len(self, example: Dict[str, Any]) -> int:
        raise NotImplementedError("BaseExecutor is an abstract class")

    def exec_result_eq(self, program_dict_1: Dict[str, Any], program_dict_2: Dict[str, Any]) -> bool:
        raise NotImplementedError("BaseExecutor is an abstract class")
    
    def get_exec_stats(self) -> Dict[str, Any]:
        stats_dict = {"cache_size": len(self.cache)}
        stats_dict.update(self.stats_dict)

        return stats_dict
    
    def process_output(self, output: str, tokenizer_eos_token: str) -> str:
        raise NotImplementedError("BaseExecutor is an abstract class")
    
    def exec_program(self, program: str, example: Dict[str, Any]) -> Tuple[int, Union[str, List, Dict]]:
        """
        returns a tuple of (if the execution result is right, and storeable execution result)
        """
        if not self.use_cache:
            return self.real_exec_program(program, example)
        else:
            if self.cache_key_func(program, example) in self.cache:
                return self.cache[self.cache_key_func(program, example)]
            else:
                exec_match, exec_result = self.real_exec_program(program, example)
                self.cache[self.cache_key_func(program, example)] = (exec_match, exec_result)
                return exec_match, exec_result

    def batch_exec_programs(self, programs: List[str], metadatas: List[Dict[str, Any]], share_metadata_n: int = 1) -> List[Any]:
        assert len(programs) == len(metadatas) * share_metadata_n

        # step 1: aggregate the same programs and reduce to pairs of (program, metadata)
        if share_metadata_n > 1:
            program_metadata_pairs = []
            for i in range(0, len(programs), share_metadata_n):
                metadata_idx = i // share_metadata_n
                distinct_example_programs = list(set(programs[i:i+share_metadata_n]))
                program_metadata_pairs.extend([(program, metadatas[metadata_idx]) for program in distinct_example_programs])
        else:
            program_metadata_pairs = [(program, metadata) for program, metadata in zip(programs, metadatas)]
        self.stats_dict["unique_program_ratio"] = len(program_metadata_pairs) / len(programs)

        # step 2: use cache to prune out the programs that have been executed before (if cache available)
        final_exec_results: List[Tuple[int, Any]] = [None for _ in range(len(program_metadata_pairs))]
        cached_indices = [] # since the execution result can be anything (including None), we need to keep track of the indices
        if self.use_cache:
            for i, (program, example) in enumerate(program_metadata_pairs):
                cache_key = self.cache_key_func(program, example)
                if cache_key in self.cache:
                    final_exec_results[i] = self.cache[cache_key]
                    cached_indices.append(i)
        cached_indices_set = set(cached_indices)  
        remaining_program_metadata_pairs = [(i, program, example) for i, (program, example) in enumerate(program_metadata_pairs) if i not in cached_indices_set]
        indices, exec_programs, exec_metadata = zip(*remaining_program_metadata_pairs)
        self.stats_dict["uncached_program_ratio"] = len(remaining_program_metadata_pairs) / len(program_metadata_pairs)
        self.stats_dict["actual_exec_ratio"] = len(remaining_program_metadata_pairs) / len(programs)
        self.stats_dict["actual_exec_n"] = len(remaining_program_metadata_pairs)

        # step 3: execute the rest of the programs
        start_time = time.time()
        with Pool(self.n_processes) as p:
            execution_results = p.map(self.__class__.real_exec_program, exec_programs, exec_metadata, timeout=10)
        execution_results = list(execution_results)
        end_time = time.time()
        self.stats_dict["exec_time"] = end_time - start_time

        # step 4: merge the results
        for i, exec_result in zip(indices, execution_results):
            final_exec_results[i] = exec_result

        if share_metadata_n > 1:
            # first build the dict of program -> execution result
            program_exec_result_dict = {self.cache_key_func(program, example): exec_result 
                                            for (program, example), exec_result in zip(program_metadata_pairs, final_exec_results)}

            # restore the original order
            return_exec_result_list = []
            for i, program in enumerate(programs):
                return_exec_result_list.append(program_exec_result_dict[self.cache_key_func(program, metadatas[i // share_metadata_n])])
            
            return return_exec_result_list
        else:
            return final_exec_results

class SpiderExecutor(BaseExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @overrides
    def cache_key_func(self, program: str, example: Dict[str, Any]) -> str:
        return example["db_id"] + program

    @overrides
    def program_len(self, program: str) -> int:
        return sql_program_len(program)

    @overrides
    def gold_program_len(self, example: Dict[str, Any]) -> int:
        return self.program_len(example["query"])

    @overrides
    def process_output(self, output: str, tokenizer_eos_token: str) -> str:
        return output.split(tokenizer_eos_token)[0].split("\n\n")[0].split(";")[0].strip()

    @staticmethod
    def real_exec_program(program: str, example: Dict[str, Any]) -> Tuple[int, Union[str, List, Dict]]:
        raw_exec_match_result = eval_exec_match(example['db_path'], example['db_path'], program, 
                                            example["query"], plug_value=False, 
                                            keep_distinct=False,
                                            progress_bar_for_each_datapoint=False)
        
        if raw_exec_match_result == -1:
            return -1, "ERROR"
        else:
            assert raw_exec_match_result in [0, 1]
            exec_match_result = int(bool(raw_exec_match_result))
            exec_result = spider_execution_pd_sql(program, example)
            if exec_result is None:
                exec_result_store = "ERROR"
            else:
                exec_result_store, _ = pd_df_to_dict(exec_result)

            return exec_match_result, exec_result_store

class WTQExecutor(SpiderExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def real_exec_program(program: str, example: Dict[str, Any]) -> Tuple[int, Union[str, List, Dict]]:
        exec_results = wtq_execution_sql(program, example)

        if exec_results is not None:
            return -1, "ERROR"
        else:
            exec_match_result = int(wtq_answer_eq(exec_results, example["original_answer"]))
            return exec_match_result, exec_results

class SpiderPythonExecutor(SpiderExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def real_exec_program(program: str, example: Dict[str, Any]) -> Tuple[int, Union[str, List, Dict]]:
        # get the table name -> dataframe dict
        conn = sqlite3.connect(example["db_path"])
        df_dicts = db_to_df_dict(conn)

        # execute the program
        exec_result = spider_execution_py(program, df_dicts)
        if "order" in example["query_toks_no_value"]:
            exec_match_result = int(spider_answer_eq(exec_result, example["answer"]))
        else:
            exec_match_result = int(spider_answer_eq(exec_result, example["answer"], True))
        
        return exec_match_result, exec_result

class WTQPythonExecutor(SpiderExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def real_exec_program(program: str, example: Dict[str, Any]) -> Tuple[int, Union[str, List, Dict]]:
        # get the table name -> dataframe dict
        conn = sqlite3.connect(example["db_path"])
        df_dicts = db_to_df_dict(conn)

        # execute the program
        exec_result = spider_execution_py(program, df_dicts)
        exec_match_result = int(wtq_answer_eq([[exec_result]], example["original_answer"]))
        
        return exec_match_result, exec_result

class MBPPExecutor(BaseExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @overrides
    def cache_key_func(self, program: str, example: Dict[str, Any]) -> str:
        return str(example["task_id"]) + program

    @overrides
    def program_len(self, program: str) -> int:
        return python_program_len(program)

    @overrides
    def gold_program_len(self, example: Dict[str, Any]) -> int:
        return self.program_len(example["code"])

    @overrides
    def process_output(self, output: str, tokenizer_eos_token: str) -> str:
        return output.split("### Task End ###")[0] # NOTE: we can't strip because python code need to maintain the indentation

    @overrides
    def exec_result_eq(self, program_dict_1: Dict[str, Any], program_dict_2: Dict[str, Any]) -> bool:
        raise NotImplementedError("BaseExecutor is an abstract class")

    @staticmethod
    def real_exec_program(program: str, example: Dict[str, Any]) -> Tuple[int, Union[str, List, Dict]]:
        # use parsability to check if the program is valid
        try:
            ast.parse(program)
        except Exception:
            return -1, "ERROR: program is not parsable"

        # then reconstruct the whole function with assertions and execute the program
        program_with_assertions = program + "\n\n" + example["test_setup_code"] + "\n\n" + "\n".join(example["test_list"])
        exec_result_with_assertions = execute(program_with_assertions, timeout=10)

        # finally check if the program passes all the tests and append the exec results
        test_exec_results = []
        for t in example["test_list"]:
            program_with_calls = program + "\n\n" + example["test_setup_code"] + "\n\n" + assertion_to_test(t)
            tracing_result = get_function_final_state(program_with_calls.strip())
            test_exec_results.append(tracing_result)

        if exec_result_with_assertions["result"] == "passed":
            return 1, test_exec_results
        else:
            return 0, test_exec_results