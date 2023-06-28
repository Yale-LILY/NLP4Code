import pandas as pd
import numpy as np


def g(df):
    l = df["A"].replace(to_replace=0, method="ffill")
    r = df["A"].replace(to_replace=0, method="bfill")
    for i in range(len(df)):
        df["A"].iloc[i] = max(l[i], r[i])
    return df


def define_test_input(args):
    if args.test_case == 1:
        index = range(14)
        data = [1, 0, 0, 2, 0, 4, 6, 8, 0, 0, 0, 0, 2, 1]
        df = pd.DataFrame(data=data, index=index, columns=["A"])
    if args.test_case == 2:
        index = range(14)
        data = [1, 0, 0, 9, 0, 2, 6, 8, 0, 0, 0, 0, 1, 7]
        df = pd.DataFrame(data=data, index=index, columns=["A"])
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
