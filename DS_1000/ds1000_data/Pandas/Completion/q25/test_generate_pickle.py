import pandas as pd
import numpy as np


def g(df, List):
    df = df[df["Date"] >= List[0]]
    df = df[df["Date"] <= List[1]]
    df["Date"] = df["Date"].dt.strftime("%d-%b-%Y %A")
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {"Date": ["2019-01-01", "2019-02-08", "2019-02-08", "2019-03-08"]}
        )
        df["Date"] = pd.to_datetime(df["Date"])
        List = ["2019-01-17", "2019-02-20"]
    return df, List


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, List = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, List), f)

    result = g(df, List)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
