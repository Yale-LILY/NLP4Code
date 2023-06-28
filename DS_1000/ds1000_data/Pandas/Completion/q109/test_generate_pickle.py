import pandas as pd
import numpy as np


def g(df1, df2):
    return pd.merge_asof(df1, df2, on="Timestamp", direction="forward")


def define_test_input(args):
    if args.test_case == 1:
        df1 = pd.DataFrame(
            {
                "Timestamp": [
                    "2019/04/02 11:00:01",
                    "2019/04/02 11:00:15",
                    "2019/04/02 11:00:29",
                    "2019/04/02 11:00:30",
                ],
                "data": [111, 222, 333, 444],
            }
        )
        df2 = pd.DataFrame(
            {
                "Timestamp": [
                    "2019/04/02 11:00:14",
                    "2019/04/02 11:00:15",
                    "2019/04/02 11:00:16",
                    "2019/04/02 11:00:30",
                    "2019/04/02 11:00:31",
                ],
                "stuff": [101, 202, 303, 404, 505],
            }
        )
        df1["Timestamp"] = pd.to_datetime(df1["Timestamp"])
        df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])
    if args.test_case == 2:
        df1 = pd.DataFrame(
            {
                "Timestamp": [
                    "2019/04/02 11:00:01",
                    "2019/04/02 11:00:15",
                    "2019/04/02 11:00:29",
                    "2019/04/02 11:00:30",
                ],
                "data": [101, 202, 303, 404],
            }
        )
        df2 = pd.DataFrame(
            {
                "Timestamp": [
                    "2019/04/02 11:00:14",
                    "2019/04/02 11:00:15",
                    "2019/04/02 11:00:16",
                    "2019/04/02 11:00:30",
                    "2019/04/02 11:00:31",
                ],
                "stuff": [111, 222, 333, 444, 555],
            }
        )
        df1["Timestamp"] = pd.to_datetime(df1["Timestamp"])
        df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])
    return df1, df2


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df1, df2 = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df1, df2), f)

    result = g(df1, df2)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
