import pandas as pd
import numpy as np


def g(df):
    df["label"] = df.Close.diff().fillna(1).gt(0).astype(int)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "DateTime": ["2000-01-04", "2000-01-05", "2000-01-06", "2000-01-07"],
                "Close": [1460, 1470, 1480, 1450],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "DateTime": ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07"],
                "Close": [1460, 1470, 1480, 1450],
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
