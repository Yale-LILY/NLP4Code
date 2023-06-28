import pandas as pd
import numpy as np


def g(df):
    df = df[sorted(df.columns.to_list())]
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["Caps", "Middle", "Lower"]
    )
    return df


def define_test_input(args):
    if args.test_case == 1:
        l = [
            ("A", "a", "1"),
            ("A", "b", "2"),
            ("B", "a", "1"),
            ("A", "b", "1"),
            ("B", "b", "1"),
            ("A", "a", "2"),
        ]
        np.random.seed(1)
        df = pd.DataFrame(np.random.randn(5, 6), columns=l)
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
