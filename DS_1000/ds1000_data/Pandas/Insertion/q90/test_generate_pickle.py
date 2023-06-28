import pandas as pd
import numpy as np


def g(df1, df2, columns_check_list):
    mask = (df1[columns_check_list] == df2[columns_check_list]).any(axis=1).values
    return mask


def define_test_input(args):
    if args.test_case == 1:
        df1 = pd.DataFrame(
            {
                "A": [1, 1, 1],
                "B": [2, 2, 2],
                "C": [3, 3, 3],
                "D": [4, 4, 4],
                "E": [5, 5, 5],
                "F": [6, 6, 6],
                "Postset": ["yes", "no", "yes"],
            }
        )
        df2 = pd.DataFrame(
            {
                "A": [1, 1, 1],
                "B": [2, 2, 2],
                "C": [3, 3, 3],
                "D": [4, 4, 4],
                "E": [5, 5, 5],
                "F": [6, 4, 6],
                "Preset": ["yes", "yes", "yes"],
            }
        )
        columns_check_list = ["A", "B", "C", "D", "E", "F"]

    return df1, df2, columns_check_list


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df1, df2, columns_check_list = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df1, df2, columns_check_list), f)

    result = g(df1, df2, columns_check_list)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
