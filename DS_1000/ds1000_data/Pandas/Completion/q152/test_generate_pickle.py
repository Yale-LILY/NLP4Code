import pandas as pd
import numpy as np


def get_relation(df, col1, col2):
    first_max = df[[col1, col2]].groupby(col1).count().max()[0]
    second_max = df[[col1, col2]].groupby(col2).count().max()[0]
    if first_max == 1:
        if second_max == 1:
            return "one-2-one"
        else:
            return "one-2-many"
    else:
        if second_max == 1:
            return "many-2-one"
        else:
            return "many-2-many"


from itertools import product


def g(df):
    result = []
    for col_i, col_j in product(df.columns, df.columns):
        if col_i == col_j:
            continue
        result.append(col_i + " " + col_j + " " + get_relation(df, col_i, col_j))
    return result


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Column1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "Column2": [4, 3, 6, 8, 3, 4, 1, 4, 3],
                "Column3": [7, 3, 3, 1, 2, 2, 3, 2, 7],
                "Column4": [9, 8, 7, 6, 5, 4, 3, 2, 1],
                "Column5": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Column1": [4, 3, 6, 8, 3, 4, 1, 4, 3],
                "Column2": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "Column3": [7, 3, 3, 1, 2, 2, 3, 2, 7],
                "Column4": [9, 8, 7, 6, 5, 4, 3, 2, 1],
                "Column5": [1, 2, 3, 4, 5, 6, 7, 8, 9],
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
