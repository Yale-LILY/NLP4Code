import pandas as pd
import numpy as np


def g(df):
    for i in df.index:
        for col in list(df):
            if type(df.loc[i, col]) == str:
                if "&AMP;" in df.loc[i, col]:
                    df.loc[i, col] = df.loc[i, col].replace("&AMP;", "&")
                    df.loc[i, col] = df.loc[i, col] + " = " + str(eval(df.loc[i, col]))
    df.replace("&AMP;", "&", regex=True)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "A": ["1 &AMP; 1", "BB", "CC", "DD", "1 &AMP; 0"],
                "B": range(5),
                "C": ["0 &AMP; 0"] * 5,
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
