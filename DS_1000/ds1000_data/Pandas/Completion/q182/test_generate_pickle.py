import pandas as pd
import numpy as np


def g(dict, df):
    df["Date"] = df["Member"].apply(lambda x: dict.get(x)).fillna(np.NAN)
    for i in range(len(df)):
        if df.loc[i, "Member"] not in dict.keys():
            df.loc[i, "Date"] = "17/8/1926"
    return df


def define_test_input(args):
    if args.test_case == 1:
        dict = {"abc": "1/2/2003", "def": "1/5/2017", "ghi": "4/10/2013"}
        df = pd.DataFrame(
            {
                "Member": ["xyz", "uvw", "abc", "def", "ghi"],
                "Group": ["A", "B", "A", "B", "B"],
                "Date": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
    if args.test_case == 2:
        dict = {"abc": "1/2/2013", "def": "1/5/2027", "ghi": "4/10/2023"}
        df = pd.DataFrame(
            {
                "Member": ["xyz", "uvw", "abc", "def", "ghi"],
                "Group": ["A", "B", "A", "B", "B"],
                "Date": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
    return dict, df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    dict, df = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((dict, df), f)

    result = g(dict, df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
