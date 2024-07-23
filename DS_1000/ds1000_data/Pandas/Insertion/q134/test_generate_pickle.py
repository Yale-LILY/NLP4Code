import pandas as pd
import numpy as np


def g(df):
    cols = list(df.filter(like="col"))
    df["index_original"] = df.groupby(cols)[cols[0]].transform("idxmax")
    for i in range(len(df)):
        i = len(df) - 1 - i
        origin = df.loc[i, "index_original"]
        if i <= origin:
            continue
        if origin == df.loc[origin, "index_original"]:
            df.loc[origin, "index_original"] = i
        df.loc[i, "index_original"] = df.loc[origin, "index_original"]
    return df[df.duplicated(subset=cols, keep="last")]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            data=[[1, 1, 2, 5], [1, 3, 4, 1], [4, 1, 2, 5], [5, 1, 4, 9], [1, 1, 2, 5]],
            columns=["val", "col1", "col2", "3col"],
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            data=[[4, 1, 2, 5], [1, 1, 2, 5], [1, 3, 4, 1], [9, 9, 1, 4], [1, 1, 2, 5]],
            columns=["val", "col1", "col2", "3col"],
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
