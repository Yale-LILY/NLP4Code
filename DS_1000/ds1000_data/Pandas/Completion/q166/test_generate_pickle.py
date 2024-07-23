import pandas as pd
import numpy as np

import numpy as np


def g(df):
    return df.groupby("a")["b"].agg([np.mean, np.std])


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "b": [12, 13, 23, 22, 23, 24, 30, 35, 55],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "a": [4, 4, 4, 5, 5, 5, 6, 6, 6],
                "b": [12, 13, 23, 22, 23, 24, 30, 35, 55],
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
