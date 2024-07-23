import pandas as pd
import numpy as np


def g(df):
    if len(df.columns) == 1:
        if df.values.size == 1:
            return df.values[0][0]
        return df.values.squeeze()
    grouped = df.groupby(df.columns[0])
    d = {k: g(t.iloc[:, 1:]) for k, t in grouped}
    return d


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "name": ["A", "A", "B", "C", "B", "A"],
                "v1": ["A1", "A2", "B1", "C1", "B2", "A2"],
                "v2": ["A11", "A12", "B12", "C11", "B21", "A21"],
                "v3": [1, 2, 3, 4, 5, 6],
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
