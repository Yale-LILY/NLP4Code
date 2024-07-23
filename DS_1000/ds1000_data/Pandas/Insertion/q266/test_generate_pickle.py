import pandas as pd
import numpy as np


def g(df):
    return (
        df.columns[df.iloc[0, :].fillna("Nan") != df.iloc[8, :].fillna("Nan")]
    ).values.tolist()


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(10)
        df = pd.DataFrame(
            np.random.randint(0, 20, (10, 10)).astype(float),
            columns=["c%d" % d for d in range(10)],
        )
        df.where(np.random.randint(0, 2, df.shape).astype(bool), np.nan, inplace=True)
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
