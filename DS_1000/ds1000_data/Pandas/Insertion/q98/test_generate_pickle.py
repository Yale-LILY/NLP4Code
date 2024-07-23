import pandas as pd
import numpy as np


def g(df):
    mask = (df.filter(like="Value").abs() > 1).any(axis=1)
    return df[mask]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "A_Name": ["AA", "BB", "CC", "DD", "EE", "FF", "GG"],
                "B_Detail": ["X1", "Y1", "Z1", "L1", "M1", "N1", "K1"],
                "Value_B": [1.2, 0.76, 0.7, 0.9, 1.3, 0.7, -2.4],
                "Value_C": [0.5, -0.7, -1.3, -0.5, 1.8, -0.8, -1.9],
                "Value_D": [-1.3, 0.8, 2.5, 0.4, -1.3, 0.9, 2.1],
            }
        )
    elif args.test_case == 2:
        df = pd.DataFrame(
            {
                "A_Name": ["AA", "BB", "CC", "DD", "EE", "FF", "GG"],
                "B_Detail": ["X1", "Y1", "Z1", "L1", "M1", "N1", "K1"],
                "Value_B": [1.2, 0.76, 0.7, 0.9, 1.3, 0.7, -2.4],
                "Value_C": [0.5, -0.7, -1.3, -0.5, 1.8, -0.8, -1.9],
                "Value_D": [-1.3, 0.8, 2.5, 0.4, -1.3, 0.9, 2.1],
                "Value_E": [1, 1, 4, -5, -1, -4, 2.1],
                "Value_F": [-1.9, 2.6, 0.8, 1.7, -1.3, 0.9, 2.1],
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
