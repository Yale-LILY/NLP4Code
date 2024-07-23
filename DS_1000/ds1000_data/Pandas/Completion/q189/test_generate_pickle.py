import pandas as pd
import numpy as np


def g(df):
    df1 = df.groupby("Date").agg(lambda x: (x % 2 == 0).sum())
    df2 = df.groupby("Date").agg(lambda x: (x % 2 == 1).sum())
    return df1, df2


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Date": ["20.07.2018", "20.07.2018", "21.07.2018", "21.07.2018"],
                "B": [10, 1, 0, 1],
                "C": [8, 0, 1, 0],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Date": ["20.07.2019", "20.07.2019", "21.07.2019", "21.07.2019"],
                "B": [10, 1, 0, 1],
                "C": [8, 0, 1, 0],
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

    result1, result2 = g(df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((result1, result2), f)
