import pandas as pd
import numpy as np


def g(df):
    df.index = df.index.set_levels(
        [df.index.levels[0], pd.to_datetime(df.index.levels[1])]
    )
    return df


def define_test_input(args):
    if args.test_case == 1:
        index = pd.MultiIndex.from_tuples(
            [("abc", "3/1/1994"), ("abc", "9/1/1994"), ("abc", "3/1/1995")],
            names=("id", "date"),
        )
        df = pd.DataFrame({"x": [100, 90, 80], "y": [7, 8, 9]}, index=index)
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
    print(result)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
