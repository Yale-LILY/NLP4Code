import pandas as pd
import numpy as np


def g(df):
    Date = list(df.index)
    Date = sorted(Date)
    half = len(list(Date)) // 2
    return max(Date, key=lambda v: Date.count(v)), Date[half]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {"value": [10000, 2000, 2000, 200, 5, 70, 200, 5, 25, 0.02, 12, 0.022]},
            index=[
                "2014-03-13",
                "2014-03-21",
                "2014-03-27",
                "2014-03-17",
                "2014-03-17",
                "2014-03-17",
                "2014-03-21",
                "2014-03-27",
                "2014-03-27",
                "2014-03-31",
                "2014-03-31",
                "2014-03-31",
            ],
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {"value": [10000, 2000, 2000, 200, 5, 70, 200, 5, 25, 0.02, 12, 0.022]},
            index=[
                "2015-03-13",
                "2015-03-21",
                "2015-03-27",
                "2015-03-17",
                "2015-03-17",
                "2015-03-17",
                "2015-03-21",
                "2015-03-27",
                "2015-03-27",
                "2015-03-31",
                "2015-03-31",
                "2015-03-31",
            ],
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

    mode_result, median_result = g(df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((mode_result, median_result), f)
