import pandas as pd
import numpy as np

import numpy as np


def g(df):
    df["state"] = np.where(
        (df["col2"] > 50) & (df["col3"] > 50),
        df["col1"],
        df[["col1", "col2", "col3"]].sum(axis=1),
    )
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "datetime": [
                    "2021-04-10 01:00:00",
                    "2021-04-10 02:00:00",
                    "2021-04-10 03:00:00",
                    "2021-04-10 04:00:00",
                    "2021-04-10 05:00:00",
                ],
                "col1": [25, 25, 25, 50, 100],
                "col2": [50, 50, 100, 50, 100],
                "col3": [50, 50, 50, 100, 100],
            }
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif args.test_case == 2:
        df = pd.DataFrame(
            {
                "datetime": [
                    "2021-04-10 01:00:00",
                    "2021-04-10 02:00:00",
                    "2021-04-10 03:00:00",
                    "2021-04-10 04:00:00",
                    "2021-04-10 05:00:00",
                ],
                "col1": [25, 10, 66, 50, 100],
                "col2": [50, 13, 100, 50, 100],
                "col3": [50, 16, 50, 100, 100],
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
