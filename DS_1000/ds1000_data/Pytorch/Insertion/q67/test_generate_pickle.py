import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn


# def define_test_input(args):
#     if args.test_case == 1:
#         a = torch.Tensor([[1, 2, 3], [1, 2, 3]])
#         b = torch.Tensor([[5, 6, 7], [5, 6, 7]])
#     elif args.test_case == 2:
#         a = torch.Tensor([[3, 2, 1], [1, 2, 3]])
#         b = torch.Tensor([[7, 6, 5], [5, 6, 7]])
#     return a, b


def ans1():
    torch.random.manual_seed(42)
    hid_dim = 32
    data = torch.randn(10, 2, 3, hid_dim)
    data = data.view(10, 2 * 3, hid_dim)
    W = torch.randn(hid_dim)
    W = W.unsqueeze(0).unsqueeze(0).expand(*data.size())
    result = torch.sum(data * W, 2)
    result = result.view(10, 2, 3)
    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    # a, b = define_test_input(args)
    #
    # if not os.path.exists("input"):
    #     os.mkdir("input")
    # with open('input/input{}.pkl'.format(args.test_case), 'wb') as f:
    #     pickle.dump((a, b), f)

    ans = ans1()
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
