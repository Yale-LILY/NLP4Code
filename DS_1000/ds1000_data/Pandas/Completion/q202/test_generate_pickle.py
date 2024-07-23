import pandas as pd
import numpy as np


def g(df):
    cols = list(df)[1:]
    for idx in df.index:
        s = 0
        cnt = 0
        for col in cols:
            if df.loc[idx, col] != 0:
                cnt = min(cnt + 1, 2)
                s = (s + df.loc[idx, col]) / cnt
            df.loc[idx, col] = s
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Name": ["Name1", "Name2", "Name3"],
                "2001": [2, 1, 0],
                "2002": [5, 4, 5],
                "2003": [0, 2, 0],
                "2004": [0, 0, 0],
                "2005": [4, 4, 0],
                "2006": [6, 0, 2],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Name": ["Name1", "Name2", "Name3"],
                "2011": [2, 1, 0],
                "2012": [5, 4, 5],
                "2013": [0, 2, 0],
                "2014": [0, 0, 0],
                "2015": [4, 4, 0],
                "2016": [6, 0, 2],
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
