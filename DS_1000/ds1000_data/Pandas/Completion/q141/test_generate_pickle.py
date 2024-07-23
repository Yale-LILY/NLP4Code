import pandas as pd
import numpy as np


def g(df):
    return pd.melt(df)


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "col1": {0: "a", 1: "b", 2: "c"},
                "col2": {0: 1, 1: 3, 2: 5},
                "col3": {0: 2, 1: 4, 2: 6},
                "col4": {0: 3, 1: 6, 2: 2},
                "col5": {0: 7, 1: 2, 2: 3},
                "col6": {0: 2, 1: 9, 2: 5},
            }
        )
        df.columns = [list("AAAAAA"), list("BBCCDD"), list("EFGHIJ")]
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "col1": {0: "a", 1: "b", 2: "c"},
                "col2": {0: 1, 1: 3, 2: 5},
                "col3": {0: 2, 1: 4, 2: 6},
                "col4": {0: 3, 1: 6, 2: 2},
                "col5": {0: 7, 1: 2, 2: 3},
                "col6": {0: 2, 1: 9, 2: 5},
            }
        )
        df.columns = [list("AAAAAA"), list("BBBCCC"), list("DDEEFF"), list("GHIJKL")]
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
