import pandas as pd
import numpy as np


def g(df):
    return (
        df.groupby("key1")["key2"]
        .apply(lambda x: x.str.endswith("e").sum())
        .reset_index(name="count")
    )


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "key1": ["a", "a", "b", "b", "a", "c"],
                "key2": ["one", "two", "gee", "two", "three", "two"],
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
