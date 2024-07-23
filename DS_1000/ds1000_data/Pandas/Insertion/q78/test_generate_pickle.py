import pandas as pd
import numpy as np


def g(df):
    return df.groupby(df.index // 4).sum()


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame({"col1": [1, 1, 4, 5, 1, 4]})
    if args.test_case == 2:
        df = pd.DataFrame({"col1": [1, 9, 2, 6, 0, 8]})
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
