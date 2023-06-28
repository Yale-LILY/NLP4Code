import pandas as pd
import numpy as np


def g(df1, df2):
    return pd.concat(
        [df1, df2.merge(df1[["id", "city", "district"]], how="left", on="id")],
        sort=False,
    ).reset_index(drop=True)


def define_test_input(args):
    if args.test_case == 1:
        df1 = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "city": ["bj", "bj", "sh", "sh", "sh"],
                "district": ["ft", "ft", "hp", "hp", "hp"],
                "date": ["2019/1/1", "2019/1/1", "2019/1/1", "2019/1/1", "2019/1/1"],
                "value": [1, 5, 9, 13, 17],
            }
        )
        df2 = pd.DataFrame(
            {
                "id": [3, 4, 5, 6, 7],
                "date": ["2019/2/1", "2019/2/1", "2019/2/1", "2019/2/1", "2019/2/1"],
                "value": [1, 5, 9, 13, 17],
            }
        )
    if args.test_case == 2:
        df1 = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "city": ["bj", "bj", "sh", "sh", "sh"],
                "district": ["ft", "ft", "hp", "hp", "hp"],
                "date": ["2019/2/1", "2019/2/1", "2019/2/1", "2019/2/1", "2019/2/1"],
                "value": [1, 5, 9, 13, 17],
            }
        )
        df2 = pd.DataFrame(
            {
                "id": [3, 4, 5, 6, 7],
                "date": ["2019/3/1", "2019/3/1", "2019/3/1", "2019/3/1", "2019/3/1"],
                "value": [1, 5, 9, 13, 17],
            }
        )
    return df1, df2


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df1, df2 = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df1, df2), f)

    result = g(df1, df2)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
