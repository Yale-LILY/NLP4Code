import pandas as pd
import numpy as np


def g(df):
    return pd.DataFrame(df.row.str.split(" ", 1).tolist(), columns=["fips", "row"])


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "row": [
                    "00000 UNITED STATES",
                    "01000 ALABAMA",
                    "01001 Autauga County, AL",
                    "01003 Baldwin County, AL",
                    "01005 Barbour County, AL",
                ]
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "row": [
                    "10000 UNITED STATES",
                    "11000 ALABAMA",
                    "11001 Autauga County, AL",
                    "11003 Baldwin County, AL",
                    "11005 Barbour County, AL",
                ]
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
