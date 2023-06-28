import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from torch import nn


def define_test_input(args):
    if args.test_case == 1:
        A_log = torch.LongTensor([0, 1, 0])
        B = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
    elif args.test_case == 2:
        A_log = torch.BoolTensor([True, False, True])
        B = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
    elif args.test_case == 3:
        A_log = torch.ByteTensor([1, 1, 0])
        B = torch.LongTensor([[999, 777, 114514], [9999, 7777, 1919810]])
    return A_log, B


def ans1(A_log, B):
    for i in range(len(A_log)):
        if A_log[i] == 1:
            A_log[i] = 0
        else:
            A_log[i] = 1
    C = B[:, A_log.bool()]
    return C


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    A_log, B = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((A_log, B), f)

    ans = ans1(A_log, B)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
