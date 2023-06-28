import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn


def define_test_input(args):
    if args.test_case == 1:
        t = torch.LongTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    elif args.test_case == 2:
        t = torch.LongTensor([[5, 6, 7], [2, 3, 4], [1, 2, 3], [7, 8, 9], [10, 11, 12]])
    return t


def ans1(t):
    result = torch.ones((t.shape[0] + 2, t.shape[1] + 2)) * -1
    result[1:-1, 1:-1] = t
    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    t = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(t, f)

    ans = ans1(t)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
