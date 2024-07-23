import pandas as pd
import numpy as np


def g(corr):
    corr_triu = corr.where(~np.tril(np.ones(corr.shape)).astype(np.bool))
    corr_triu = corr_triu.stack()
    return corr_triu[corr_triu > 0.3]


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(10)
        df = pd.DataFrame(np.random.rand(10, 5))
        corr = df.corr()

    return corr


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    corr = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(corr, f)

    result = g(corr)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
