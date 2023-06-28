import pandas as pd
import numpy as np


def g(df):
    return df.mask((df == df.min()).cumsum().astype(bool))[::-1].idxmax()


def define_test_input(args):
    if args.test_case == 1:
        a = np.array(
            [
                [1.0, 0.9, 1.0],
                [0.9, 0.9, 1.0],
                [0.8, 1.0, 0.5],
                [1.0, 0.3, 0.2],
                [1.0, 0.2, 0.1],
                [0.9, 1.0, 1.0],
                [1.0, 0.9, 1.0],
                [0.6, 0.9, 0.7],
                [1.0, 0.9, 0.8],
                [1.0, 0.8, 0.9],
            ]
        )
        idx = pd.date_range("2017", periods=a.shape[0])
        df = pd.DataFrame(a, index=idx, columns=list("abc"))
    if args.test_case == 2:
        a = np.array(
            [
                [1.0, 0.9, 1.0],
                [0.9, 0.9, 1.0],
                [0.8, 1.0, 0.5],
                [1.0, 0.3, 0.2],
                [1.0, 0.2, 0.1],
                [0.9, 1.0, 1.0],
                [0.9, 0.9, 1.0],
                [0.6, 0.9, 0.7],
                [1.0, 0.9, 0.8],
                [1.0, 0.8, 0.9],
            ]
        )
        idx = pd.date_range("2022", periods=a.shape[0])
        df = pd.DataFrame(a, index=idx, columns=list("abc"))
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
