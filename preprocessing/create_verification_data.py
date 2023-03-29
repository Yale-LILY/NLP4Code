import os
import re
import sys
import json

from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
from numpy.random import choice
from collections import Counter
from transformers import AutoTokenizer, CodeGenTokenizer
from execution.executors import SpiderExecutor, WTQExecutor, MathExecutor, MBPPExecutor, BaseExecutor

class CodexTokenizer:
    """a fake tokenizer for the Codex model"""
    def batch_decode(self, token_lists: List[List[str]]):
        return ["".join(tokens) for tokens in token_lists]

def bounds_analysis(examples: List[Dict[str, Any]], executor):
    oracle = 0

    execution_rate = 0
    unique_percentage = 0
    random_correct = 0
    greedy_correct = 0
    most_confident_correct = 0
    normalized_most_confident_correct = 0
    most_freq_correct = 0
    majority_vote_correct = 0 

    error_filtered_random_correct = 0
    error_filtered_most_confident_correct = 0
    error_filtered_normalized_most_confident_correct = 0
    error_filtered_most_freq_correct = 0
    error_filtered_majority_vote_correct = 0

    if executor is None:
        print(f"Majority vote not enabled since no executor is given")

    def get_majority_vote_program(program_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        program_counts = []
        for program_dict in program_dicts:
            if len(program_counts) == 0:
                program_counts.append((program_dict, program_dict["program_count"]))
                continue
            
            idx = -1
            for i, (ref_program_dict, _) in enumerate(program_counts):
                if executor.exec_result_eq(program_dict, ref_program_dict):
                    idx = i
                    break
            
            if idx == -1:
                program_counts.append((program_dict, program_dict["program_count"]))
            else:
                program_counts[idx] = (program_counts[idx][0], program_counts[idx][1] + program_dict["program_count"])
        
        return sorted(program_counts, key=lambda x: x[1], reverse=True)[0][0]

    for example in examples:
        # measure all the selections without error filtering
        program_dicts = example["generated_programs"]
        oracle += int(sum([program_dict['exec_match'] for program_dict in program_dicts]) > 0)
        if "generated_program" in example:
            greedy_correct += example["generated_program"]["exec_match"]

        execution_rate += sum([int(not isinstance(program_dict['exec_result'], str)) for program_dict in program_dicts]) / len(program_dicts)
        unique_percentage += len(program_dicts) / sum([p["program_count"] for p in program_dicts])
        random_correct += sum([program_dict['exec_match'] for program_dict in program_dicts]) / len(program_dicts)
        if 'gen_prob' in program_dicts[0]:
            most_confident_correct += sorted(program_dicts, key=lambda x: x['gen_prob'], reverse=True)[0]['exec_match']
            normalized_most_confident_correct += sorted(program_dicts, key=lambda x: x['norm_gen_prob'], reverse=True)[0]['exec_match'] 
        most_freq_correct += sorted(program_dicts, key=lambda x: x['program_count'], reverse=True)[0]['exec_match']
        if executor is not None:
            majority_vote_correct += get_majority_vote_program(program_dicts)['exec_match']

        # measure all the selections with error filtering
        error_filtered_program_dicts = list(filter(lambda x: not isinstance(x['exec_result'], str), example["generated_programs"]))
        if len(error_filtered_program_dicts) != 0:
            error_filtered_random_correct += sum([program_dict['exec_match'] for program_dict in error_filtered_program_dicts]) / len(error_filtered_program_dicts)
            if 'gen_prob' in program_dicts[0]:
                error_filtered_most_confident_correct += sorted(error_filtered_program_dicts, key=lambda x: x['gen_prob'], reverse=True)[0]['exec_match']
                error_filtered_normalized_most_confident_correct += sorted(error_filtered_program_dicts, key=lambda x: x['norm_gen_prob'], reverse=True)[0]['exec_match']
            error_filtered_most_freq_correct += sorted(error_filtered_program_dicts, key=lambda x: x['program_count'], reverse=True)[0]['exec_match']
            if executor is not None:
                error_filtered_majority_vote_correct += get_majority_vote_program(error_filtered_program_dicts)['exec_match']

    print(f"Oracle: {oracle / len(examples)}")

    print(f"Execution Rate: {execution_rate / len(examples)}")
    print(f"Unique Percentage: {unique_percentage / len(examples)}")
    print(f"Random: {random_correct / len(examples)}")
    print(f"Greedy: {greedy_correct / len(examples)}")
    print(f"Most Confident: {most_confident_correct / len(examples)}")
    print(f"Normalized Most Confident: {normalized_most_confident_correct / len(examples)}")
    print(f"Most Frequent: {most_freq_correct / len(examples)}")
    if executor is not None:
        print(f"Majority Vote: {majority_vote_correct / len(examples)}")

    print(f"Error Filtered Random: {error_filtered_random_correct / len(examples)}")
    print(f"Error Filtered Most Confident: {error_filtered_most_confident_correct / len(examples)}")
    print(f"Error Filtered Normalized Most Confident: {error_filtered_normalized_most_confident_correct / len(examples)}")
    print(f"Error Filtered Most Frequent: {error_filtered_most_freq_correct / len(examples)}")
    if executor is not None:
        print(f"Error Filtered Majority Vote: {error_filtered_majority_vote_correct / len(examples)}")

def process_mbpp_program(code: str, exec_result: Any, exec_match: bool, metadata: Dict[str, Any]=None) -> Dict[str, Any]:
    program_dict = {"code": code,
                    "exec_result": exec_result, 
                    "exec_match": exec_match,
                    "program_count": 1}
    
    return program_dict

def process_spider_program(code: str, exec_result: Any, exec_match: bool, metadata: Dict[str, Any]=None) -> Dict[str, Any]:
    # process the raw code outputs
    processed_code = code.split("\n\n")[0].strip()
    processed_code = processed_code.replace("\"", "'")
    lower_code = re.sub(r"\b(?<!')(\w+)(?!')\b", lambda match: match.group(1).lower(), processed_code) 
    
    # construct the program dictionary
    program_dict = {"code": processed_code, "lower_code": lower_code, 
                    "exec_result": exec_result, 
                    "exec_match": exec_match,
                    "program_count": 1}
    
    return program_dict

def process_gsmath_program(code: str, exec_result: Any, exec_match: bool, metadata: Dict[str, Any]=None) -> Dict[str, Any]:
    # process the raw code outputs
    # process the raw code outputs
    processed_code = "\n".join(list(filter(lambda x: len(x.strip()) > 0, code.split("\n"))))
    processed_code = processed_code.split("\n\n")[0].strip()
    processed_code = processed_code.replace("\"", "'")
    lower_code = re.sub(r"\b(?<!')(\w+)(?!')\b", lambda match: match.group(1).lower(), code) 
    
    # construct the program dictionary
    program_dict = {"code": processed_code, "lower_code": lower_code, 
                    "exec_result": exec_result, 
                    "exec_match": exec_match,
                    "program_count": 1}
    
    return program_dict

def binary_search_for_gen_prob(program: str, tokens: Union[List[str], List[int]], 
                               tokenizer: AutoTokenizer, decode_settings: Dict[str, Any]) -> Tuple[float, float]:
    start = 0
    end = len(tokens)
    minimum_seq = tokenizer.batch_decode([tokens], **decode_settings)[0]
    while start < end - 1:
        new_end = (start + end) // 2
        decoded_result = tokenizer.batch_decode([tokens[:new_end]], **decode_settings)[0]
        if program in decoded_result:
            minimum_seq = decoded_result
            end = new_end
        else:
            start = new_end
    
    if len(minimum_seq.strip()) - len(program.strip()) > 3:
        print(f"found minimum seq significantly longer than the program: \n{repr(minimum_seq)}\n{repr(program)}\n")
    
    return end

def get_gen_prob(program: str, tokens: Union[List[str], List[int]], token_probs: List[float], 
                 tokenizer: AutoTokenizer, decode_settings: Dict[str, Any]) -> Tuple[float, float]:
    if program not in tokenizer.batch_decode([tokens], **decode_settings)[0]:
        print(f"Cannot find the program \n{repr(program)}\n in the tokens \n{repr(tokenizer.batch_decode([tokens], **decode_settings)[0])}\n")
        end_idx = len(tokens)
    else:
        end_idx = binary_search_for_gen_prob(program, tokens, tokenizer, decode_settings)
    return sum(token_probs[:end_idx]), sum(token_probs[:end_idx]) / end_idx

def combine_unicode_bytes_in_tokens(tokens: List[str]) -> List[str]:
    result_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "<|endoftext|>":
            break
        if token.startswith("bytes:"):
            byte_str = token[6:]
            while i+1 < len(tokens) and tokens[i+1].startswith("bytes:"):
                i += 1
                byte_str += tokens[i][6:]

            hex_list = byte_str.split("\\x") # NOTE: first element is not a hex
            assert len(hex_list[-1]) == 2
            for j in range(4): 
                # this is because the last unicode might be cut off and unicode is at most 4 bytes
                try:
                    combined_hex_str = "".join(hex_list[1:len(hex_list)-j])
                    decoded_str = hex_list[0] + bytes.fromhex(combined_hex_str).decode("utf-8")
                    break
                except UnicodeDecodeError:
                    continue

            result_tokens.append(decoded_str)
        else:
            result_tokens.append(token)
        i += 1

    return result_tokens

def set_exec_info_for_program_dict(program_dict: Dict[str, Any], exec_status: int, exec_result: Any) -> Dict[str, Any]:
    program_dict["exec_result"] = exec_result
    program_dict["exec_acc"] = float(exec_status == 1)
    program_dict["exec_rate"] = float(exec_status >= 0)
    return program_dict

def get_executor(dataset_name: str, n_processes: int = 20, use_cache: bool = True) -> BaseExecutor: 
    if dataset_name == "spider":
        return SpiderExecutor(n_processes=n_processes, use_cache=use_cache)
    elif dataset_name == "gsm":
        return MathExecutor(n_processes=n_processes, use_cache=use_cache)
    elif dataset_name == "squall":
        return WTQExecutor(n_processes=n_processes, use_cache=use_cache)
    elif dataset_name == "mbpp":
        return MBPPExecutor(n_processes=n_processes, use_cache=use_cache)
    else:
        raise ValueError(f"Cannot find the executor for {dataset_name}")

def get_model_dataset_name_from_file(file_name: str) -> Tuple[str, str]:
    # find the model name
    if "codex" in file_name.lower():
        model_name = "codex"
    elif "incoder" in file_name.lower():
        model_name = "incoder"
    elif "codegen" in file_name.lower():
        model_name = "codegen"
    else:
        raise ValueError(f"Cannot find the model name from the file name {file_name}")
    
    # find the dataset name from {gsm, spider, squall, mbpp}
    if "gsm" in file_name.lower():
        dataset_name = "gsm"
    elif "spider" in file_name.lower():
        dataset_name = "spider"
    elif "squall" in file_name.lower():
        dataset_name = "squall"
    elif "mbpp" in file_name.lower():
        dataset_name = "mbpp"
    else:
        raise ValueError(f"Cannot find the dataset name from the file name {file_name}")
    
    return model_name, dataset_name

def create_verification_examples(all_results: List[Dict[str, Any]], model_name: str, dataset_name: str, force_execute: bool):
    # model specific processing
    decode_settings = {}
    if model_name == "codex":
        model_name = "codex"
        tokenizer = CodexTokenizer()
    elif model_name == "incoder":
        model_name = "incoder"
        tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-6B")
        decode_settings = {"clean_up_tokenization_spaces": False}
    elif model_name == "codegen":
        model_name = "codegen"
        tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-16B-multi")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # init the executor for the specific dataset
    executor = get_executor(dataset_name)

    # more dataset specific processing
    if dataset_name == "spider":
        gold_program_annotation = True
        get_gold_program = lambda x: x["query"]
        get_program_dict = process_spider_program
    elif dataset_name == "squall":
        gold_program_annotation = True
        get_gold_program = lambda x: x["query"]
        get_program_dict = process_spider_program
    elif dataset_name == "mbpp":
        gold_program_annotation = True
        get_gold_program = lambda x: x["code"]
        get_program_dict = process_mbpp_program
    elif dataset_name == "gsm":
        gold_program_annotation = False
        get_gold_program = None
        get_program_dict = process_gsmath_program
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    # collect all the gold program and execute them in batch (since we need them to be added during training)
    if gold_program_annotation:
        gold_program_exec_results = executor.batch_exec_programs([get_gold_program(x["metadata"]) for x in all_results],
                                                                [x["metadata"] for x in all_results])
        print(f"gold program exec results match rate: {sum([int(x[0] == 1) for x in gold_program_exec_results]) / len(gold_program_exec_results)}")
    
    # in case some generate programs don't have execution result
    missing_idx_original_idx_dict = dict()
    missing_exec_program_list = []
    missing_exec_metadata_list = []
    program_in_example = -1
    for i, example in enumerate(all_results):
        if force_execute or example["generated_program"]["exec_result"] == "TBD":
            missing_idx_original_idx_dict[len(missing_idx_original_idx_dict)] = i
            missing_exec_program_list.extend([example["generated_program"]["program"]] + [x["program"] for x in example["generated_k_programs"]])
            missing_exec_metadata_list.append(example["metadata"])
            program_in_example = 1 + len(example["generated_k_programs"])
    if len(missing_exec_program_list) > 0:
        print(f"Missing {len(missing_exec_program_list)} programs from {len(missing_idx_original_idx_dict)} examples, executing them now")
    
        # execute the missing programs and add them to the results
        missing_exec_results = executor.batch_exec_programs(missing_exec_program_list, missing_exec_metadata_list, share_metadata_n=program_in_example)
        for j, missing_exec_result in enumerate(missing_exec_results):
            exec_status, exec_result = missing_exec_result
            original_idx = missing_idx_original_idx_dict[j // program_in_example]
            program_idx = j % program_in_example
            if program_idx == 0:
                set_exec_info_for_program_dict(all_results[original_idx]["generated_program"], exec_status, exec_result)
            else:
                set_exec_info_for_program_dict(all_results[original_idx]["generated_k_programs"][program_idx - 1], exec_status, exec_result)

    # process all examples
    processed_results = []
    for i, result in tqdm(enumerate(all_results), total=len(all_results)):
        example_dict = {"metadata": result["metadata"]}

        # process the gold program dict
        if gold_program_annotation:
            gold_exec_status, gold_exec_result = gold_program_exec_results[i]
            exec_info_program_dict = set_exec_info_for_program_dict({}, gold_exec_status, gold_exec_result) # use a dummy dict to get the execution info processed
            gold_program_dict = get_program_dict(get_gold_program(result["metadata"]), 
                                                 exec_info_program_dict["exec_result"], 
                                                 exec_info_program_dict["exec_acc"],
                                                 result["metadata"])
            example_dict["gold_program"] = gold_program_dict

        # process the greedy program dict
        greedy_result = result["generated_program"]
        greedy_program_dict = get_program_dict(greedy_result["program"], greedy_result["exec_result"], 
                                             greedy_result["exec_acc"], result["metadata"])
        example_dict["generated_program"] = greedy_program_dict

        # deal with the unicode bytes in the tokens for codex
        if model_name == "codex" and "generation_tokens" in result:
            for i in range(len(result["generation_tokens"])):
                result["generation_tokens"][i] = combine_unicode_bytes_in_tokens(result["generation_tokens"][i])

        # process the k program dicts without duplication
        program_set = dict()
        generated_k_program_dicts = []
        for j, result_dict in enumerate(result["generated_k_programs"]):
            result_program = result_dict["program"]
            if result_program in program_set:
                # only update the program count and then skip
                duplicate_program_idx = program_set[result_program]
                generated_k_program_dicts[duplicate_program_idx]["program_count"] += 1
                continue

            # add the program dict to the set
            program_set[result_program] = len(generated_k_program_dicts)
            generated_program_dict = get_program_dict(result_program, result_dict["exec_result"], 
                                                      result_dict["exec_acc"], result["metadata"])
            
            # add the generation probability if available
            if "gen_prob" in result_dict:
                # this is from last run's analysis
                generated_program_dict["gen_prob"] = result_dict["gen_prob"]
                generated_program_dict["norm_gen_prob"] = result_dict["norm_gen_prob"]
            elif "generation_tokens" in result and "generation_probs" in result:                                          
                gen_prob, norm_gen_prob = get_gen_prob(result_program, result["generation_tokens"][j], 
                                                    result["generation_probs"][j], tokenizer, decode_settings)
                generated_program_dict["gen_prob"] = gen_prob
                generated_program_dict["norm_gen_prob"] = norm_gen_prob
            generated_k_program_dicts.append(generated_program_dict)

        # add the k program dicts to the example dict
        example_dict["generated_programs"] = generated_k_program_dicts
        processed_results.append(example_dict)
    
    return processed_results
    

def main():
    """
    Two modes are supported:
    1. analysis existing verification file: -a [-f] <verification_file_name> [new_verification_file_name]
        [Optional -f] to force execute all programs
        <verification_file_name> is the verification file *.jsonl to be analyzed
        [Optional new_verification_file_name] is the new verification file *.jsonl to be created
    2. creation and analysis: [-f] <verification_file_name>  <few_shot_result_dirs> ...
        [Optional -f] to force execute all programs
        <verification_file_name> is the verification file *.jsonl to be created
        <few_shot_result_dirs>... is a list of directories containing the few-shot results
    """

    args = sys.argv[1:]

    if args[0] == "-a":
        ##########################
        ### analysis only mode ###
        ##########################

        downsample_n = None
        if args[1] == "-d":
            downsample_n = int(args[2])
            file_name_args = args[3:]
        elif args[1] == "-f":
            force_execute = True
            file_name_args = args[2:]
        else:
            force_execute = False
            file_name_args = args[1:]
        
        # get the file names
        source_file_name = file_name_args[0]
        if len(file_name_args) > 1:
            new_file_name = file_name_args[1]
        else:
            new_file_name = None

        # read and get the model and dataset names
        with open(source_file_name, "r") as f:
            results = [json.loads(s) for s in f.readlines()]
        model_name, dataset_name = get_model_dataset_name_from_file(source_file_name)

        if downsample_n is not None:
            # simply do downsampling
            print(f"Downsampling {downsample_n} programs per example from {source_file_name} ...")

            for example in results:
                generated_programs = example["generated_programs"]
                program_counts = [p["program_count"] for p in generated_programs]
                sample_probs = [p / sum(program_counts) for p in program_counts]
                downsampled_index_counts = Counter(choice(len(generated_programs), size=downsample_n, replace=True, p=sample_probs))

                new_generated_programs = []
                for i, count in downsampled_index_counts.items():
                    generated_programs[i]["program_count"] = count
                    new_generated_programs.append(generated_programs[i])
                
                assert sum([p["program_count"] for p in new_generated_programs]) == downsample_n
                example["generated_programs"] = new_generated_programs

            processed_examples = results

            # save the results to a new file if specified
            if new_file_name is not None:
                with open(new_file_name, "w") as f:
                    for example in processed_examples:
                        f.write(json.dumps(example) + "\n")

            # now do the analysis
            bounds_analysis(processed_examples, get_executor(dataset_name))
        elif force_execute:
            # convert the results to the original format and go through the verification file creation process again
            original_results = []
            for result in results:
                original_dict = {
                    "generated_k_programs": [],
                    "metadata": result["metadata"],
                }

                if dataset_name == "gsm":
                    if "original_answer" in original_dict["metadata"]:
                        # re-parse the answer as previous parsing may be wrong
                        original_dict["metadata"]["answer"] = float(original_dict["metadata"]["original_answer"].split("\n####")[-1].strip().replace(",", ""))
                    elif isinstance(original_dict["metadata"]["answer"], str):
                        original_dict["metadata"]["original_answer"] = original_dict["metadata"]["answer"]
                        original_dict["metadata"]["answer"] = float(original_dict["metadata"]["answer"].split("\n####")[-1].strip().replace(",", ""))

                # add the k programs
                for program_dict in result["generated_programs"]:
                    for i in range(program_dict["program_count"]):
                        # reverse the program aggregation process
                        original_dict["generated_k_programs"].append({"program": program_dict["code"],
                                                                      "gen_prob": program_dict["gen_prob"],
                                                                      "norm_gen_prob": program_dict["norm_gen_prob"]})

                # add the gold and greedy program if available
                if "generated_program" in result:
                    original_dict["generated_program"] = {"program": result["generated_program"]["code"]}
                else:
                    # has to have generated_program to be able to re-execute
                    original_dict["generated_program"] = {"program": result["generated_programs"][0]["code"]}

                if "gold_program" in result:
                    original_dict["gold_program"] = {"program": result["gold_program"]["code"]}

                original_results.append(original_dict)
            
            # get the re-executed verification data
            processed_examples = create_verification_examples(original_results, model_name, dataset_name, force_execute=True)

            # save the results to a new file if specified
            if new_file_name is not None:
                with open(new_file_name, "w") as f:
                    for example in processed_examples:
                        f.write(json.dumps(example) + "\n")

            # now do the analysis
            bounds_analysis(processed_examples, get_executor(dataset_name))
        else:
            # only perform the analysis
            bounds_analysis(results, get_executor(dataset_name))
    else:
        ###########################
        ### create and analysis ###
        ###########################
        if args[0] == "-f":
            force_execute = True
            print("Force executing all generated programs...")
            llm_result_dirs = args[2:]
            output_file = args[1]
        else:
            force_execute = False
            llm_result_dirs = args[1:]
            output_file = args[0]
        
        # gather the prediction files
        pred_files = []
        for llm_result_dir in llm_result_dirs:
            file_exist = False
            for file in os.listdir(llm_result_dir):
                if file.endswith(".jsonl"):
                    file_exist = True
                    assert file.startswith("predictions")
                    pred_files.append(os.path.join(llm_result_dir, file))
            if not file_exist:
                print(f"WARN: No prediction file found in {llm_result_dir}")

        # get the model and dataset name
        model_name, dataset_name = get_model_dataset_name_from_file(llm_result_dirs[0])
        
        # read and merge all the prediction results
        all_results = []
        for pred_file in pred_files:
            with open(pred_file, "r") as f:
                all_results.extend([json.loads(s) for s in f.readlines()])
        print(f"Loaded {len(all_results)} results from {len(pred_files)} files")

        # process the examples
        processed_results = create_verification_examples(all_results, model_name, dataset_name, force_execute)

        # write the processed results to the output file
        with open(output_file, "w") as f:
            for result in processed_results:
                f.write(json.dumps(result) + "\n")
        
        # do analysis after the results are written in case error occurs in the analysis
        bounds_analysis(processed_results, get_executor(dataset_name))
        
if __name__ == "__main__":
    main()