import pandas as pd
import numpy as np


def g(df):
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Caps", "Lower"])
    return df


def define_test_input(args):
    if args.test_case == 1:
        l = [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")]
        np.random.seed(1)
        df = pd.DataFrame(np.random.randn(5, 4), columns=l)
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
