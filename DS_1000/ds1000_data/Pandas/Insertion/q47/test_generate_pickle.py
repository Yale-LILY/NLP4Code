import pandas as pd
import numpy as np


def g(df, thresh):
    return df[lambda x: x["value"] >= thresh].append(
        df[lambda x: x["value"] < thresh].sum().rename("X")
    )


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {"lab": ["A", "B", "C", "D", "E", "F"], "value": [50, 35, 8, 5, 1, 1]}
        )
        df = df.set_index("lab")
        thresh = 6
    if args.test_case == 2:
        df = pd.DataFrame(
            {"lab": ["A", "B", "C", "D", "E", "F"], "value": [50, 35, 8, 5, 1, 1]}
        )
        df = df.set_index("lab")
        thresh = 9
    return df, thresh


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, thresh = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, thresh), f)

    result = g(df, thresh)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
