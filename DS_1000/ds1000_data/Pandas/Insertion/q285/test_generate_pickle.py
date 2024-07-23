import pandas as pd
import numpy as np


def g(df):
    df["frequent"] = df.mode(axis=1)
    for i in df.index:
        df.loc[i, "freq_count"] = (df.iloc[i] == df.loc[i, "frequent"]).sum() - 1
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "bit1": [0, 2, 4],
                "bit2": [0, 2, 0],
                "bit3": [3, 0, 4],
                "bit4": [3, 0, 4],
                "bit5": [0, 2, 4],
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
