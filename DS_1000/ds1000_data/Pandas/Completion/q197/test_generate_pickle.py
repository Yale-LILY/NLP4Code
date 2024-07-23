import pandas as pd
import numpy as np


def g(df):
    df["new"] = df.apply(lambda p: sum(not q.isalpha() for q in p["str"]), axis=1)
    df["new"] = df["new"].replace(0, np.NAN)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame({"str": ["Aa", "Bb", "?? ?", "###", "{}xxa;"]})
    if args.test_case == 2:
        df = pd.DataFrame({"str": ["Cc", "Dd", "!! ", "###%", "{}xxa;"]})
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
