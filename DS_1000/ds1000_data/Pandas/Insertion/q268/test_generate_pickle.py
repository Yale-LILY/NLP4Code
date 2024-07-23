import pandas as pd
import numpy as np


def g(df):
    return pd.Series(df["Value"].values, index=df["Date"])


def define_test_input(args):
    if args.test_case == 1:
        dates = ["2016-1-{}".format(i) for i in range(1, 21)]
        values = [i for i in range(20)]
        data = {"Date": dates, "Value": values}
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
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
