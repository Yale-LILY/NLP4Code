import pandas as pd
import numpy as np


def g(df):
    l = []
    for i in range(2 * (len(df) // 5) + (len(df) % 5) // 3 + 1):
        l.append(0)
    for i in range(len(df)):
        idx = 2 * (i // 5) + (i % 5) // 3
        if i % 5 < 3:
            l[idx] += df["col1"].iloc[i]
        elif i % 5 == 3:
            l[idx] = df["col1"].iloc[i]
        else:
            l[idx] = (l[idx] + df["col1"].iloc[i]) / 2
    return pd.DataFrame({"col1": l})


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame({"col1": [2, 1, 3, 1, 0, 2, 1, 3, 1]})
    if args.test_case == 2:
        df = pd.DataFrame({"col1": [1, 9, 2, 6, 0, 8, 1, 7, 1]})
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
