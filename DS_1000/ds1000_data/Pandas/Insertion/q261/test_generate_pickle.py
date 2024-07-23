import pandas as pd
import numpy as np


def g(df):
    df["TIME"] = pd.to_datetime(df["TIME"])
    df["TIME"] = df["TIME"].dt.strftime("%d-%b-%Y %a %T")
    df["RANK"] = df.groupby("ID")["TIME"].rank(ascending=False)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "ID": ["01", "01", "01", "02", "02"],
                "TIME": [
                    "2018-07-11 11:12:20",
                    "2018-07-12 12:00:23",
                    "2018-07-13 12:00:00",
                    "2019-09-11 11:00:00",
                    "2019-09-12 12:00:00",
                ],
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
