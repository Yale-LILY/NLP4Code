import pandas as pd
import numpy as np


def g(df, row_list, column_list):
    return df[column_list].iloc[row_list].mean(axis=0)


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "a": [1, 1, 1, 1],
                "b": [2, 2, 1, 0],
                "c": [3, 3, 1, 0],
                "d": [0, 4, 6, 0],
                "q": [5, 5, 1, 0],
            }
        )
        row_list = [0, 2, 3]
        column_list = ["a", "b", "d"]
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "a": [1, 1, 1, 1],
                "b": [2, 2, 1, 0],
                "c": [3, 3, 1, 0],
                "d": [0, 4, 6, 0],
                "q": [5, 5, 1, 0],
            }
        )
        row_list = [0, 1, 3]
        column_list = ["a", "c", "q"]
    return df, row_list, column_list


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, row_list, column_list = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, row_list, column_list), f)

    result = g(df, row_list, column_list)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
