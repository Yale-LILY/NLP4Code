import pandas as pd
import numpy as np


def g(df):
    cols = list(df)
    Mode = df.mode(axis=1)
    df["frequent"] = df["bit1"].astype(object)
    for i in df.index:
        df.at[i, "frequent"] = []
    for i in df.index:
        for col in list(Mode):
            if pd.isna(Mode.loc[i, col]) == False:
                df.at[i, "frequent"].append(Mode.loc[i, col])
        df.at[i, "frequent"] = sorted(df.at[i, "frequent"])
        df.loc[i, "freq_count"] = (df[cols].iloc[i] == df.loc[i, "frequent"][0]).sum()
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
                "bit6": [3, 0, 5],
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
