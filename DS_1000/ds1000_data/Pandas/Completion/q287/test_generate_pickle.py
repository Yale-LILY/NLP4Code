import pandas as pd
import numpy as np


def g(df):
    df["bar"] = pd.to_numeric(df["bar"], errors="coerce")
    res = df.groupby(["id1", "id2"])[["foo", "bar"]].mean()
    return res


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "foo": [8, 5, 3, 4, 7, 9, 5, 7],
                "id1": [1, 1, 1, 1, 1, 1, 1, 1],
                "bar": ["NULL", "NULL", "NULL", 1, 3, 4, 2, 3],
                "id2": [1, 1, 1, 2, 2, 3, 3, 1],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "foo": [18, 5, 3, 4, 17, 9, 5, 7],
                "id1": [1, 1, 1, 1, 1, 1, 1, 1],
                "bar": ["NULL", "NULL", "NULL", 11, 3, 4, 2, 3],
                "id2": [1, 1, 1, 2, 2, 3, 3, 1],
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
