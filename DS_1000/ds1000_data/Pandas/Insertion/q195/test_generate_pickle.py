import pandas as pd
import numpy as np


def g(df):
    return (
        df.join(
            pd.DataFrame(
                df.var2.str.split(",", expand=True)
                .stack()
                .reset_index(level=1, drop=True),
                columns=["var2 "],
            )
        )
        .drop("var2", 1)
        .rename(columns=str.strip)
        .reset_index(drop=True)
    )


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            [["A", "Z,Y"], ["B", "X"], ["C", "W,U,V"]],
            index=[1, 2, 3],
            columns=["var1", "var2"],
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            [["A", "Z,Y,X"], ["B", "W"], ["C", "U,V"]],
            index=[1, 2, 3],
            columns=["var1", "var2"],
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
