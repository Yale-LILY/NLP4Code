import pandas as pd
import numpy as np


def g(df):
    F = {}
    cnt = 0
    for i in range(len(df)):
        if df["a"].iloc[i] not in F.keys():
            cnt += 1
            F[df["a"].iloc[i]] = cnt
        df.loc[i, "a"] = F[df.loc[i, "a"]]
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "name": ["Aaron", "Aaron", "Aaron", "Brave", "Brave", "David"],
                "a": [3, 3, 3, 4, 3, 5],
                "b": [5, 6, 6, 6, 6, 1],
                "c": [7, 9, 10, 0, 1, 4],
            }
        )
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
