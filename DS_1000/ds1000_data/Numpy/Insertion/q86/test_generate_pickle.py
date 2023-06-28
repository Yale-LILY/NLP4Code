import numpy as np
import pandas as pd
import torch
import copy


def define_test_input(args):
    if args.test_case == 1:
        a = torch.ones(5)
    elif args.test_case == 2:
        a = torch.tensor([1, 1, 4, 5, 1, 4])

    return a


def ans1(_a):
    a = _a
    a_np = a.numpy()

    return a_np


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(copy.deepcopy(a))

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
