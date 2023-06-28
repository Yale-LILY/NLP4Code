import pandas as pd
import numpy as np


def g(df):
    df.dt = pd.to_datetime(df.dt)
    return (
        df.set_index(["dt", "user"])
        .unstack(fill_value=0)
        .asfreq("D", fill_value=0)
        .stack()
        .sort_index(level=1)
        .reset_index()
    )


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "user": ["a", "a", "b", "b"],
                "dt": ["2016-01-01", "2016-01-02", "2016-01-05", "2016-01-06"],
                "val": [1, 33, 2, 1],
            }
        )
        df["dt"] = pd.to_datetime(df["dt"])
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "user": ["c", "c", "d", "d"],
                "dt": ["2016-02-01", "2016-02-02", "2016-02-05", "2016-02-06"],
                "val": [1, 33, 2, 1],
            }
        )
        df["dt"] = pd.to_datetime(df["dt"])
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
