import pandas as pd
import numpy as np


def g(df):
    df.loc[df["name"].str.split().str.len() >= 3, "middle_name"] = (
        df["name"].str.split().str[1:-1]
    )
    for i in range(len(df)):
        if len(df.loc[i, "name"].split()) >= 3:
            l = df.loc[i, "name"].split()[1:-1]
            s = l[0]
            for j in range(1, len(l)):
                s += " " + l[j]
            df.loc[i, "middle_name"] = s
    df.loc[df["name"].str.split().str.len() >= 2, "last_name"] = (
        df["name"].str.split().str[-1]
    )
    df.loc[df["name"].str.split().str.len() >= 2, "name"] = (
        df["name"].str.split().str[0]
    )
    df.rename(columns={"name": "first name"}, inplace=True)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {"name": ["Jack Fine", "Kim Q. Danger", "Jane 114 514 Smith", "Zhongli"]}
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
