import pandas as pd
import numpy as np


def g(df):
    return df.query("closing_price < 99 or closing_price > 101")


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(2)
        df = pd.DataFrame({"closing_price": np.random.randint(95, 105, 10)})

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
