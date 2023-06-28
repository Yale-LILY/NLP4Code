import pandas as pd
import numpy as np


def g(df):
    return df.groupby("cokey").apply(pd.DataFrame.sort_values, "A")


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "cokey": [11168155, 11168155, 11168155, 11168156, 11168156],
                "A": [18, 0, 56, 96, 0],
                "B": [56, 18, 96, 152, 96],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "cokey": [155, 155, 155, 156, 156],
                "A": [18, 0, 56, 96, 0],
                "B": [56, 18, 96, 152, 96],
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
