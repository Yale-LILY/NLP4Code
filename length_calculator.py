import json
import argparse
import numpy as np
from transformers import GPT2Tokenizer
# from execution.executors import sql_program_len, python_program_len
from ds1000.ds1000 import DS1000Dataset

def sql_program_len(code: str) -> int:
    """ return the length of the sql query """
    return len(list(filter(lambda x: not len(x.strip()) == 0, code.split())))

def python_program_len(code: str) -> int:
    """ return the length of the python program """
    return len(list(filter(lambda x: not x.startswith("#") and not len(x.strip()) == 0, code.split("\n"))))


TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')

def load_ds1000():
    ds_data = DS1000Dataset("ds1000/ds1000_data") # loads all questions into RAM
    return ds_data
    

def parse_arg() -> tuple[str, str]:
    parser = argparse.ArgumentParser(description='Process a file at the specified path')

    parser.add_argument('path', type=str, help='the path of the file to process')

    choices = ['mbpp', 'gsmath', 'spider', 'ds1000']
    parser.add_argument('--dataset', type=str, default='mbpp', choices=choices, help='the name of the dataset to use')
    
    args = parser.parse_args()

    path = args.path
    dataset = args.dataset

    return path, dataset

def generate_length(lines, dataset):
    NL_lengths = []

    program_lengths = []

    extra_info_lengths = []

    if dataset == "ds1000":
        ds_data = load_ds1000()
        for key in ds_data.data.keys():
            for data in ds_data[key]:
                program_lengths.append(python_program_len(data['reference_code']))
                extra_info_lengths.append(python_program_len(data['prompt']))
                print(data['ans'])

    for line in lines:
        data = json.loads(line)

        if dataset == "mbpp":
            NL_lengths.append(len(TOKENIZER.encode(data['text'])))
            program_lengths.append(python_program_len(data['code']))

        elif dataset == "gsmath":
            question_length = len(TOKENIZER.encode(data['question']))
            NL_lengths.append(question_length)

        elif dataset == "spider":
            NL_lengths.append(len(TOKENIZER.encode(data['question'])))
            program_lengths.append(sql_program_len(data['query']))
            extra_info_lengths.append(len(TOKENIZER.encode(str(data['db_table_headers']))))
                
            
    if NL_lengths:
        print("Average length of natural language input: {:.2f}".format(np.mean(NL_lengths)))
        print("90% percentile of length of natural language input: {:.2f}".format(np.percentile(NL_lengths, 90)))
        print("99% percentile of length of natural language input: {:.2f}".format(np.percentile(NL_lengths, 99)))
    if program_lengths:
        print("Average length of reference program: {:.2f}".format(np.mean(program_lengths)))
    if extra_info_lengths:
        print("Average length of extra information: {:.2f}".format(np.mean(extra_info_lengths)))

def main():
    path, dataset = parse_arg()

    with open(path, 'r') as f:
        lines = f.readlines()

    generate_length(lines, dataset)

main()