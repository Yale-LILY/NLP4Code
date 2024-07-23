import pandas as pd
import numpy as np


def g(df, filter_list):
    return df.query("Category != @filter_list")


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {"Category": ["Foo", "Bar", "Cho", "Foo"], "Index": [1, 2, 3, 4]}
        )
        filter_list = ["Foo", "Bar"]
    if args.test_case == 2:
        df = pd.DataFrame(
            {"Category": ["Foo", "Bar", "Cho", "Foo", "Bar"], "Index": [1, 2, 3, 4, 5]}
        )
        filter_list = ["Foo", "Bar"]
    return df, filter_list


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, filter_list = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, filter_list), f)

    result = g(df, filter_list)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
