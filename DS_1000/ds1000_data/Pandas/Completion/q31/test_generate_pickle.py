import pandas as pd
import numpy as np


def g(df):
    return df.add_prefix("X")


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame({"HeaderA": [476], "HeaderB": [4365], "HeaderC": [457]})
    if args.test_case == 2:
        df = pd.DataFrame({"HeaderD": [114], "HeaderF": [4365], "HeaderG": [514]})
    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    result = g(df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
