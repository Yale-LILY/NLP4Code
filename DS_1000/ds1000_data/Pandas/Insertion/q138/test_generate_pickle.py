import pandas as pd
import numpy as np


def g(df):
    return df[df.groupby(["Sp", "Value"])["count"].transform(max) == df["count"]]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Sp": ["MM1", "MM1", "MM1", "MM2", "MM2", "MM2", "MM4", "MM4", "MM4"],
                "Value": ["S1", "S1", "S3", "S3", "S4", "S4", "S2", "S2", "S2"],
                "Mt": ["a", "n", "cb", "mk", "bg", "dgd", "rd", "cb", "uyi"],
                "count": [3, 2, 5, 8, 10, 1, 2, 2, 7],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Sp": ["MM2", "MM2", "MM4", "MM4", "MM4"],
                "Mt": ["S4", "S4", "S2", "S2", "S2"],
                "Value": ["bg", "dgd", "rd", "cb", "uyi"],
                "count": [10, 1, 2, 8, 8],
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
