import pandas as pd
import numpy as np


def g(df):
    for i in range(len(df)):
        tot = 0
        if i != 0:
            if df.loc[i, "UserId"] == df.loc[i - 1, "UserId"]:
                continue
        for j in range(len(df)):
            if df.loc[i, "UserId"] == df.loc[j, "UserId"]:
                tot += 1
        l = int(0.2 * tot)
        dfupdate = df.iloc[i : i + tot].sample(l, random_state=0)
        dfupdate.Quantity = 0
        df.update(dfupdate)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "UserId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                "ProductId": [1, 4, 7, 4, 2, 1, 1, 4, 7, 4, 2, 1, 1, 4, 7],
                "Quantity": [6, 1, 3, 2, 7, 2, 6, 1, 3, 2, 7, 2, 6, 1, 3],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "UserId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                "ProductId": [6, 1, 3, 2, 7, 2, 6, 1, 3, 2, 7, 2, 6, 1, 3],
                "Quantity": [1, 4, 7, 4, 2, 1, 1, 4, 7, 4, 2, 1, 1, 4, 7],
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
