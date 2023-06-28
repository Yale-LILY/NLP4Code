import pandas as pd
import numpy as np


def g(df):
    s = ""
    for c in df.columns:
        s += "---- %s ---" % c
        s += "\n"
        s += str(df[c].value_counts())
        s += "\n"
    return s


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            data=[[34, "null", "mark"], [22, "null", "mark"], [34, "null", "mark"]],
            columns=["id", "temp", "name"],
            index=[1, 2, 3],
        )
    elif args.test_case == 2:
        df = pd.DataFrame(
            data=[[11, "null", "mark"], [14, "null", "mark"], [51, "null", "mark"]],
            columns=["id", "temp", "name"],
            index=[1, 2, 3],
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
