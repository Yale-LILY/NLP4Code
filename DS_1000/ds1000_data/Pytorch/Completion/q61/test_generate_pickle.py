import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn


def define_test_input(args):
    if args.test_case == 1:
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        input = torch.from_numpy(X[42]).float()
    return input


def ans1(input):
    MyNet = torch.nn.Sequential(
        torch.nn.Linear(4, 15),
        torch.nn.Sigmoid(),
        torch.nn.Linear(15, 3),
    )
    MyNet.load_state_dict(torch.load("my_model.pt"))

    """
    training part
    """
    # X, Y = load_iris(return_X_y=True)
    # lossFunc = torch.nn.CrossEntropyLoss()
    # opt = torch.optim.Adam(MyNet.parameters(), lr=0.001)
    # for batch in range(0, 50):
    #     for i in range(len(X)):
    #         x = MyNet(torch.from_numpy(X[i]).float()).reshape(1, 3)
    #         y = torch.tensor(Y[i]).long().unsqueeze(0)
    #         loss = lossFunc(x, y)
    #         loss.backward()
    #         opt.step()
    #         opt.zero_grad()
    #         # print(x.grad)
    #         # print(loss)
    #     # print(loss)
    output = MyNet(input)
    probs = torch.nn.functional.softmax(output.reshape(1, 3), dim=1)
    confidence_score, classes = torch.max(probs, 1)
    return confidence_score


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    input = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(input, f)

    ans = ans1(input)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
