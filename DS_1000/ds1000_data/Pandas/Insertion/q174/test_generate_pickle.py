import pandas as pd
import numpy as np

import numpy as np


def g(s):
    result = s.iloc[np.lexsort([s.index, s.values])].reset_index(drop=False)
    result.columns = ["index", 1]
    return result


def define_test_input(args):
    if args.test_case == 1:
        s = pd.Series(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.98, 0.93],
            index=[
                "146tf150p",
                "havent",
                "home",
                "okie",
                "thanx",
                "er",
                "anything",
                "lei",
                "nite",
                "yup",
                "thank",
                "ok",
                "where",
                "beerage",
                "anytime",
                "too",
                "done",
                "645",
                "tick",
                "blank",
            ],
        )
    if args.test_case == 2:
        s = pd.Series(
            [1, 1, 1, 1, 1, 1, 1, 0.86, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.98, 0.93],
            index=[
                "146tf150p",
                "havent",
                "homo",
                "okie",
                "thanx",
                "er",
                "anything",
                "lei",
                "nite",
                "yup",
                "thank",
                "ok",
                "where",
                "beerage",
                "anytime",
                "too",
                "done",
                "645",
                "tick",
                "blank",
            ],
        )
    return s


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    s = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(s, f)

    result = g(s)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
