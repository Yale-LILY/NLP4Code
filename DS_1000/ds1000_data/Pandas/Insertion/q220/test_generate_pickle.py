import pandas as pd
import numpy as np


def g(df):
    df["SOURCE_NAME"] = df["SOURCE_NAME"].str.rsplit("_", 1).str.get(0)
    return df


def define_test_input(args):
    if args.test_case == 1:
        strs = [
            "Stackoverflow_1234",
            "Stack_Over_Flow_1234",
            "Stackoverflow",
            "Stack_Overflow_1234",
        ]
        df = pd.DataFrame(data={"SOURCE_NAME": strs})
    if args.test_case == 2:
        strs = [
            "Stackoverflow_4321",
            "Stack_Over_Flow_4321",
            "Stackoverflow",
            "Stack_Overflow_4321",
        ]
        df = pd.DataFrame(data={"SOURCE_NAME": strs})
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
