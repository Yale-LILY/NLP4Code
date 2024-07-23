import pandas as pd
import numpy as np

import math


def g(df):
    return df.join(df.apply(lambda x: 1 / x).add_prefix("inv_")).replace(math.inf, 0)


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    if args.test_case == 2:
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
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
