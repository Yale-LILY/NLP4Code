import pandas as pd
import numpy as np

import numpy as np


def g(df):
    family = np.where((df["SibSp"] + df["Parch"]) >= 1, "Has Family", "No Family")
    return df.groupby(family)["Survived"].mean()


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Survived": [0, 1, 1, 1, 0],
                "SibSp": [1, 1, 0, 1, 0],
                "Parch": [0, 0, 0, 0, 1],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Survived": [1, 0, 0, 0, 1],
                "SibSp": [0, 0, 1, 0, 1],
                "Parch": [1, 1, 1, 1, 0],
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
