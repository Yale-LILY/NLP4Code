import pandas as pd
import numpy as np

import yaml


def g(df):
    df.message = df.message.replace(["\[", "\]"], ["{", "}"], regex=True).apply(
        yaml.safe_load
    )
    df1 = pd.DataFrame(df.pop("message").values.tolist(), index=df.index)
    result = pd.concat([df, df1], axis=1)
    result = result.replace("", "none")
    result = result.replace(np.nan, "none")
    return result


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "name": ["matt", "james", "adam"],
                "status": ["active", "active", "inactive"],
                "number": [12345, 23456, 34567],
                "message": [
                    "[job:  , money: none, wife: none]",
                    "[group: band, wife: yes, money: 10000]",
                    "[job: none, money: none, wife:  , kids: one, group: jail]",
                ],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "name": ["matt", "james", "adam"],
                "status": ["active", "active", "inactive"],
                "number": [12345, 23456, 34567],
                "message": [
                    "[job:  , money: 114514, wife: none, kids: one, group: jail]",
                    "[group: band, wife: yes, money: 10000]",
                    "[job: none, money: none, wife:  , kids: one, group: jail]",
                ],
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
