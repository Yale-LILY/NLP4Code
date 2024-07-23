import pandas as pd
import numpy as np


def g(df, filt):
    df = df[filt[df.index.get_level_values("a")].values]
    return df[filt[df.index.get_level_values("b")].values]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "b": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "c": range(9),
            }
        ).set_index(["a", "b"])
        filt = pd.Series({1: True, 2: False, 3: True})
    elif args.test_case == 2:
        df = pd.DataFrame(
            {
                "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "b": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "c": range(9),
            }
        ).set_index(["a", "b"])
        filt = pd.Series({1: True, 2: True, 3: False})
    return df, filt


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, filt = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, filt), f)

    result = g(df, filt)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
