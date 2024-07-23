import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import pandas as pd
import numpy as np
import scipy.stats as stats


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            [[1, 5, 2], [2, 4, 4], [3, 3, 1], [4, 2, 2], [5, 1, 4]],
            columns=["A", "B", "C"],
            index=[1, 2, 3, 4, 5],
        )
    elif args.test_case == 2:
        df = pd.DataFrame(
            [[1, 3, 2], [2, 5, 4], [2, 3, 1], [1, 2, 2], [5, 8, 4]],
            columns=["A", "B", "C"],
            index=[1, 2, 3, 4, 5],
        )

    return df


def ans1(_a):
    df = _a
    import itertools as IT

    for col1, col2 in IT.combinations(df.columns, 2):

        def tau(idx):
            B = df[[col1, col2]].iloc[idx]
            return stats.kendalltau(B[col1], B[col2])[0]

        df[col1 + col2] = (
            pd.Series(np.arange(len(df)), index=df.index).rolling(3).apply(tau)
        )

    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(a.copy())

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
