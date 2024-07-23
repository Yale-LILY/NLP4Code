import pandas as pd
import numpy as np


def g(df):
    df[["number", "time"]] = df.duration.str.extract(r"(\d+)\s*(.*)", expand=True)
    df["time_days"] = df["time"].replace(
        ["year", "month", "week", "day"], [365, 30, 7, 1], regex=True
    )
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {"duration": ["7 year", "2day", "4 week", "8 month"]},
            index=list(range(1, 5)),
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {"duration": ["2 year", "6day", "8 week", "7 month"]},
            index=list(range(1, 5)),
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
