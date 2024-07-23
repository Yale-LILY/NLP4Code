import pandas as pd
import numpy as np


def g(df):
    df["dogs"] = df["dogs"].apply(lambda x: round(x, 2) if str(x) != "<NA>" else x)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            [
                (0.21, 0.3212),
                (0.01, 0.61237),
                (0.66123, 0.03),
                (0.21, 0.18),
                (pd.NA, 0.18),
            ],
            columns=["dogs", "cats"],
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            [
                (0.215, 0.3212),
                (0.01, 0.11237),
                (0.36123, 0.03),
                (0.21, 0.18),
                (pd.NA, 0.18),
            ],
            columns=["dogs", "cats"],
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
