import pandas as pd
import numpy as np


def g(df):
    return pd.pivot_table(
        df, values=["D", "E"], index=["B"], aggfunc={"D": np.max, "E": np.min}
    )


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(1)
        df = pd.DataFrame(
            {
                "A": ["one", "one", "two", "three"] * 6,
                "B": ["A", "B", "C"] * 8,
                "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
                "D": np.random.randn(24),
                "E": np.random.randn(24),
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
