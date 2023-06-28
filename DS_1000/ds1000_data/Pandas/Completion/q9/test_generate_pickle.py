import pandas as pd
import numpy as np


def g(df):
    return df.loc[(df["keep_if_dup"] == "Yes") | ~df["url"].duplicated(keep="last")]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "url": ["A.com", "A.com", "A.com", "B.com", "B.com", "C.com", "B.com"],
                "keep_if_dup": ["Yes", "Yes", "No", "No", "No", "No", "Yes"],
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
