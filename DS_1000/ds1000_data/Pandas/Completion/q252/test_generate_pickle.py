import pandas as pd
import numpy as np


def g(df):
    df = df.codes.apply(pd.Series)
    cols = list(df)
    for i in range(len(cols)):
        cols[i] += 1
    df.columns = cols
    return df.add_prefix("code_")


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "codes": [
                    [71020],
                    [77085],
                    [36416],
                    [99213, 99287],
                    [99233, 99233, 99233],
                ]
            }
        )
    elif args.test_case == 2:
        df = pd.DataFrame(
            {
                "codes": [
                    [71020, 71011],
                    [77085],
                    [99999, 36415],
                    [99213, 99287],
                    [99233, 99232, 99234],
                ]
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
