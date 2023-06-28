import pandas as pd
import numpy as np


def g(C, D):
    return (
        pd.concat([C, D])
        .drop_duplicates("A", keep="first")
        .sort_values(by=["A"])
        .reset_index(drop=True)
    )


def define_test_input(args):
    if args.test_case == 1:
        C = pd.DataFrame({"A": ["AB", "CD", "EF"], "B": [1, 2, 3]})
        D = pd.DataFrame({"A": ["CD", "GH"], "B": [4, 5]})
    if args.test_case == 2:
        D = pd.DataFrame({"A": ["AB", "CD", "EF"], "B": [1, 2, 3]})
        C = pd.DataFrame({"A": ["CD", "GH"], "B": [4, 5]})
    return C, D


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    C, D = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((C, D), f)

    result = g(C, D)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
