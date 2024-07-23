import pandas as pd
import numpy as np


def g(df):
    df_out = df.stack()
    df_out.index = df_out.index.map("{0[1]}_{0[0]}".format)
    return df_out.to_frame().T


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            columns=["A", "B", "C", "D", "E"],
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], columns=["A", "B", "C", "D", "E"]
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
