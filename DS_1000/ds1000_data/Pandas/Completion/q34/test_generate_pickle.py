import pandas as pd
import numpy as np


def g(df):
    return df.groupby("group").agg(
        lambda x: x.head(1) if x.dtype == "object" else x.sum()
    )


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B"],
                "group_color": ["green", "green", "green", "blue", "blue"],
                "val1": [5, 2, 3, 4, 5],
                "val2": [4, 2, 8, 5, 7],
                "val3": [1, 1, 4, 5, 1],
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
