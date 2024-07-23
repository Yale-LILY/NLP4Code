import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import io


import pandas as pd
import io
from scipy import integrate


def define_test_input(args):
    if args.test_case == 1:
        string = """
Time                      A
2017-12-18-19:54:40   -50187.0
2017-12-18-19:54:45   -60890.5
2017-12-18-19:54:50   -28258.5
2017-12-18-19:54:55    -8151.0
2017-12-18-19:55:00    -9108.5
2017-12-18-19:55:05   -12047.0
2017-12-18-19:55:10   -19418.0
2017-12-18-19:55:15   -50686.0
2017-12-18-19:55:20   -57159.0
2017-12-18-19:55:25   -42847.0
"""
        df = pd.read_csv(io.StringIO(string), sep="\s+")

    return df


def ans1(_a):
    df = _a
    df.Time = pd.to_datetime(df.Time, format="%Y-%m-%d-%H:%M:%S")
    df = df.set_index("Time")
    integral_df = df.rolling("25S").apply(integrate.trapz)

    return integral_df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(a)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
