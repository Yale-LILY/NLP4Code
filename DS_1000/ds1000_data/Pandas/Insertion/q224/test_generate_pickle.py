import pandas as pd
import numpy as np


def g(df):
    idx = df["Column_x"].index[df["Column_x"].isnull()]
    total_nan_len = len(idx)
    first_nan = (total_nan_len * 3) // 10
    middle_nan = (total_nan_len * 3) // 10
    df.loc[idx[0:first_nan], "Column_x"] = 0
    df.loc[idx[first_nan : first_nan + middle_nan], "Column_x"] = 0.5
    df.loc[idx[first_nan + middle_nan : total_nan_len], "Column_x"] = 1
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Column_x": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Column_x": [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
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
