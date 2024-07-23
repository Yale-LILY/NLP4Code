import pandas as pd
import numpy as np


def g(df):
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df.sort_values(by="datetime", inplace=True)
    df["datetime"] = df["datetime"].dt.strftime("%d-%b-%Y %T")
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "datetime": [
                    "2015-12-01 00:00:00-06:00",
                    "2015-12-02 00:01:00-06:00",
                    "2015-12-03 00:00:00-06:00",
                ]
            }
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif args.test_case == 2:
        df = pd.DataFrame(
            {
                "datetime": [
                    "2016-12-02 00:01:00-06:00",
                    "2016-12-01 00:00:00-06:00",
                    "2016-12-03 00:00:00-06:00",
                ]
            }
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
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
