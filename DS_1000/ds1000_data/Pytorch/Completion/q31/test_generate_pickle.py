import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn


def define_test_input(args):
    if args.test_case == 1:
        torch.random.manual_seed(42)
        a = torch.rand((10, 1000, 23))
        lengths = torch.randint(1000, (10,))
    return a, lengths


def ans1(a, lengths):
    for i_batch in range(10):
        a[i_batch, : lengths[i_batch], :] = 2333
    return a


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a, lengths = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((a, lengths), f)

    ans = ans1(a, lengths)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
