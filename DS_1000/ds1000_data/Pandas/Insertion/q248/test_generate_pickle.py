import pandas as pd
import numpy as np


def g(df, s):
    spike_cols = [col for col in df.columns if s in col and col != s]
    return spike_cols


def define_test_input(args):
    if args.test_case == 1:
        data = {
            "spike-2": [1, 2, 3],
            "hey spke": [4, 5, 6],
            "spiked-in": [7, 8, 9],
            "no": [10, 11, 12],
            "spike": [13, 14, 15],
        }
        df = pd.DataFrame(data)
        s = "spike"
    return df, s


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, s = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, s), f)

    result = g(df, s)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
