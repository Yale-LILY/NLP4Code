import pandas as pd
import numpy as np


def g(df):
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
    y = df["Date"].dt.year
    m = df["Date"].dt.month

    df["Count_d"] = df.groupby("Date")["Date"].transform("size")
    df["Count_m"] = df.groupby([y, m])["Date"].transform("size")
    df["Count_y"] = df.groupby(y)["Date"].transform("size")
    df["Count_Val"] = df.groupby(["Date", "Val"])["Val"].transform("size")
    return df


def define_test_input(args):
    if args.test_case == 1:
        d = {
            "Date": [
                "1/1/18",
                "1/1/18",
                "1/1/18",
                "2/1/18",
                "3/1/18",
                "1/2/18",
                "1/3/18",
                "2/1/19",
                "3/1/19",
            ],
            "Val": ["A", "A", "B", "C", "D", "A", "B", "C", "D"],
        }
        df = pd.DataFrame(data=d)
    if args.test_case == 2:
        d = {
            "Date": [
                "1/1/19",
                "1/1/19",
                "1/1/19",
                "2/1/19",
                "3/1/19",
                "1/2/19",
                "1/3/19",
                "2/1/20",
                "3/1/20",
            ],
            "Val": ["A", "A", "B", "C", "D", "A", "B", "C", "D"],
        }
        df = pd.DataFrame(data=d)
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
