import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import cross_val_predict, StratifiedKFold


def define_test_input(args):
    if args.test_case == 1:
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    return X, y


def ans1(X, y):
    cv = StratifiedKFold(5).split(X, y)
    logreg = LogisticRegression(random_state=42)
    proba = cross_val_predict(logreg, X, y, cv=cv, method="predict_proba")
    return proba


def ans2(X, y):
    cv = StratifiedKFold(5).split(X, y)
    logreg = LogisticRegression(random_state=42)
    proba = []
    for train, test in cv:
        logreg.fit(X[train], y[train])
        proba.append(logreg.predict_proba(X[test]))
    return proba


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    X, y = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((X, y), f)

    ans = ans1(X.copy(), y.copy()), ans2(X.copy(), y.copy())
    # np.testing.assert_allclose(ans[0],ans[1])
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
