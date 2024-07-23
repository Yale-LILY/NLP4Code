import pandas as pd
import numpy as np


def g(df):
    cols = list(df)[:2] + list(df)[-1:1:-1]
    df = df.loc[:, cols]
    return (
        df.set_index(["Country", "Variable"])
        .rename_axis(["year"], axis=1)
        .stack()
        .unstack("Variable")
        .reset_index()
    )


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Country": ["Argentina", "Argentina", "Brazil", "Brazil"],
                "Variable": ["var1", "var2", "var1", "var2"],
                "2000": [12, 1, 20, 0],
                "2001": [15, 3, 23, 1],
                "2002": [18, 2, 25, 2],
                "2003": [17, 5, 29, 2],
                "2004": [23, 7, 31, 3],
                "2005": [29, 5, 32, 3],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Country": ["Argentina", "Argentina", "Brazil", "Brazil"],
                "Variable": ["var1", "var2", "var1", "var2"],
                "2000": [12, 1, 20, 0],
                "2001": [15, 1, 23, 1],
                "2002": [18, 4, 25, 2],
                "2003": [17, 5, 29, 2],
                "2004": [23, 1, 31, 3],
                "2005": [29, 4, 32, 3],
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
