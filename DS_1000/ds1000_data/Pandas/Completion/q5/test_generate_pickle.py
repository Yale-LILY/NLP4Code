import pandas as pd
import numpy as np


def g(df):
    for col in df.columns:
        vc = df[col].value_counts()
        if col == "Qu1":
            df[col] = df[col].apply(lambda x: x if vc[x] >= 3 else "other")
        else:
            df[col] = df[col].apply(lambda x: x if vc[x] >= 2 else "other")
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Qu1": [
                    "apple",
                    "potato",
                    "cheese",
                    "banana",
                    "cheese",
                    "banana",
                    "cheese",
                    "potato",
                    "egg",
                ],
                "Qu2": [
                    "sausage",
                    "banana",
                    "apple",
                    "apple",
                    "apple",
                    "sausage",
                    "banana",
                    "banana",
                    "banana",
                ],
                "Qu3": [
                    "apple",
                    "potato",
                    "sausage",
                    "cheese",
                    "cheese",
                    "potato",
                    "cheese",
                    "potato",
                    "egg",
                ],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Qu1": [
                    "sausage",
                    "banana",
                    "apple",
                    "apple",
                    "apple",
                    "sausage",
                    "banana",
                    "banana",
                    "banana",
                ],
                "Qu2": [
                    "apple",
                    "potato",
                    "sausage",
                    "cheese",
                    "cheese",
                    "potato",
                    "cheese",
                    "potato",
                    "egg",
                ],
                "Qu3": [
                    "apple",
                    "potato",
                    "cheese",
                    "banana",
                    "cheese",
                    "banana",
                    "cheese",
                    "potato",
                    "egg",
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
