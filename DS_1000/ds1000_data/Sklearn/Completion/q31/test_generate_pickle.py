import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def define_test_input(args):
    if args.test_case == 1:
        iris = load_iris()
    return iris.data, iris.target


def ans1(data, target):
    pipe = Pipeline(steps=[("select", SelectKBest(k=2)), ("clf", LogisticRegression())])
    select_out = pipe.named_steps["select"].fit_transform(data, target)
    return select_out


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    data, target = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((data, target), f)

    ans = ans1(data, target)
    np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
