import pandas as pd
import numpy as np


def g(df, list_of_my_columns):
    df["Avg"] = df[list_of_my_columns].mean(axis=1)
    df["Min"] = df[list_of_my_columns].min(axis=1)
    df["Max"] = df[list_of_my_columns].max(axis=1)
    df["Median"] = df[list_of_my_columns].median(axis=1)
    return df


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(10)
        data = {}
        for i in [chr(x) for x in range(65, 91)]:
            data["Col " + i] = np.random.randint(1, 100, 10)
        df = pd.DataFrame(data)
        list_of_my_columns = ["Col A", "Col E", "Col Z"]

    return df, list_of_my_columns


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, list_of_my_columns = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, list_of_my_columns), f)

    result = g(df, list_of_my_columns)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
