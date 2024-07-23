import pandas as pd
import numpy as np


def g(df):
    return df.groupby("user")[["time", "amount"]].apply(lambda x: x.values.tolist())


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "user": [1, 1, 2, 2, 3],
                "time": [20, 10, 11, 18, 15],
                "amount": [10.99, 4.99, 2.99, 1.99, 10.99],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "user": [1, 1, 1, 2, 2, 3],
                "time": [20, 10, 30, 11, 18, 15],
                "amount": [10.99, 4.99, 16.99, 2.99, 1.99, 10.99],
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
