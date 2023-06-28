import pandas as pd
import numpy as np

import numpy as np


def g(df):
    df["#1"] = np.roll(df["#1"], shift=1)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "#1": [11.6985, 43.6431, 54.9089, 63.1225, 72.4399],
                "#2": [126.0, 134.0, 130.0, 126.0, 120.0],
            },
            index=[
                "1980-01-01",
                "1980-01-02",
                "1980-01-03",
                "1980-01-04",
                "1980-01-05",
            ],
        )
    elif args.test_case == 2:
        df = pd.DataFrame(
            {"#1": [45, 51, 14, 11, 14], "#2": [126.0, 134.0, 130.0, 126.0, 120.0]},
            index=[
                "1980-01-01",
                "1980-01-02",
                "1980-01-03",
                "1980-01-04",
                "1980-01-05",
            ],
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
