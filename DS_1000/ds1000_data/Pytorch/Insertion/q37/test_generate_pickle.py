import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn


def define_test_input(args):
    if args.test_case == 1:
        t = torch.tensor([[-0.2, 0.3], [-0.5, 0.1], [-0.4, 0.2]])
        idx = np.array([1, 0, 1], dtype=np.int32)
    elif args.test_case == 2:
        t = torch.tensor([[-0.2, 0.3], [-0.5, 0.1], [-0.4, 0.2]])
        idx = np.array([1, 1, 0], dtype=np.int32)
    elif args.test_case == 3:
        t = torch.tensor([[-22.2, 33.3], [-55.5, 11.1], [-44.4, 22.2]])
        idx = np.array([1, 1, 0], dtype=np.int32)
    return t, idx


def ans1(t, idx):
    idxs = torch.from_numpy(idx).long().unsqueeze(1)
    # or   torch.from_numpy(idxs).long().view(-1,1)
    result = t.gather(1, idxs).squeeze(1)
    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    t, idx = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((t, idx), f)

    ans = ans1(t, idx)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
