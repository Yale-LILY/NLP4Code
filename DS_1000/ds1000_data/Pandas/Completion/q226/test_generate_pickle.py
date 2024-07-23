import pandas as pd
import numpy as np


def g(a, b):
    return pd.DataFrame(
        np.rec.fromarrays((a.values, b.values)).tolist(),
        columns=a.columns,
        index=a.index,
    )


def define_test_input(args):
    if args.test_case == 1:
        a = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=["one", "two"])
        b = pd.DataFrame(np.array([[5, 6], [7, 8]]), columns=["one", "two"])
    if args.test_case == 2:
        b = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=["one", "two"])
        a = pd.DataFrame(np.array([[5, 6], [7, 8]]), columns=["one", "two"])
    return a, b


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a, b = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((a, b), f)

    result = g(a, b)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
