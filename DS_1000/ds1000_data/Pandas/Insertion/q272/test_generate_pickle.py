import pandas as pd
import numpy as np


def g(df):
    for i in df.index:
        if str(df.loc[i, "dogs"]) != "<NA>" and str(df.loc[i, "cats"]) != "<NA>":
            df.loc[i, "dogs"] = round(df.loc[i, "dogs"], 2)
            df.loc[i, "cats"] = round(df.loc[i, "cats"], 2)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            [
                (0.21, 0.3212),
                (0.01, 0.61237),
                (0.66123, pd.NA),
                (0.21, 0.18),
                (pd.NA, 0.188),
            ],
            columns=["dogs", "cats"],
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            [
                (pd.NA, 0.3212),
                (0.01, 0.61237),
                (0.66123, pd.NA),
                (0.21, 0.18),
                (pd.NA, 0.188),
            ],
            columns=["dogs", "cats"],
        )
    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=2)
    args = parser.parse_args()

    df = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    result = g(df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
