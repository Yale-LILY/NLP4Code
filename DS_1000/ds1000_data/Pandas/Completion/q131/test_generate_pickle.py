import pandas as pd
import numpy as np


def g(df):
    df["index_original"] = df.groupby(["col1", "col2"]).col1.transform("idxmax")
    for i in range(len(df)):
        i = len(df) - 1 - i
        origin = df.loc[i, "index_original"]
        if i <= origin:
            continue
        if origin == df.loc[origin, "index_original"]:
            df.loc[origin, "index_original"] = i
        df.loc[i, "index_original"] = df.loc[origin, "index_original"]
    return df[df.duplicated(subset=["col1", "col2"], keep="last")]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            data=[[1, 2], [3, 4], [1, 2], [1, 4], [1, 2]], columns=["col1", "col2"]
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            data=[[1, 1], [3, 1], [1, 4], [1, 9], [1, 6]], columns=["col1", "col2"]
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
