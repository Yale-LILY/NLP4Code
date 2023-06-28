import pandas as pd
import numpy as np


def g(df_a, df_b):
    return df_a[["EntityNum", "foo"]].merge(
        df_b[["EntityNum", "a_col"]], on="EntityNum", how="left"
    )


def define_test_input(args):
    if args.test_case == 1:
        df_a = pd.DataFrame(
            {"EntityNum": [1001.01, 1002.02, 1003.03], "foo": [100, 50, 200]}
        )
        df_b = pd.DataFrame(
            {
                "EntityNum": [1001.01, 1002.02, 1003.03],
                "a_col": ["alice", "bob", "777"],
                "b_col": [7, 8, 9],
            }
        )
    if args.test_case == 2:
        df_a = pd.DataFrame(
            {"EntityNum": [1001.01, 1002.02, 1003.03], "foo": [100, 50, 200]}
        )
        df_b = pd.DataFrame(
            {
                "EntityNum": [1001.01, 1002.02, 1003.03],
                "a_col": ["666", "bob", "alice"],
                "b_col": [7, 8, 9],
            }
        )
    return df_a, df_b


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df_a, df_b = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df_a, df_b), f)

    result = g(df_a, df_b)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
