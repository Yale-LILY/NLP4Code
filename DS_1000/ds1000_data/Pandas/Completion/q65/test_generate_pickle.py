import pandas as pd
import numpy as np


def g(df):
    df = (
        df.set_index(["user", "someBool"])
        .stack()
        .reset_index(name="value")
        .rename(columns={"level_2": "date"})
    )
    return df[["user", "date", "value", "someBool"]]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "user": ["u1", "u2", "u3"],
                "01/12/15": [100, 200, -50],
                "02/12/15": [300, -100, 200],
                "someBool": [True, False, True],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "user": ["u1", "u2", "u3"],
                "01/10/22": [100, 200, -50],
                "02/10/22": [300, -100, 200],
                "someBool": [True, False, True],
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
