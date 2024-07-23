import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def define_test_input(args):
    if args.test_case == 1:
        X, y = make_classification(
            n_samples=30, n_features=4, n_redundant=0, random_state=42
        )
    elif args.test_case == 2:
        X, y = make_classification(
            n_samples=30, n_features=4, n_redundant=0, random_state=24
        )
    return X, y


def ans1(X_train, y_train):
    X_test = X_train
    param_grid = {
        "base_estimator__max_depth": [1, 2, 3, 4, 5],
        "max_samples": [0.05, 0.1, 0.2, 0.5],
    }
    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
    bc = BaggingClassifier(
        dt, n_estimators=20, max_samples=0.5, max_features=0.5, random_state=42
    )
    clf = GridSearchCV(bc, param_grid)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    return proba


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    X_train, y_train = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((X_train, y_train), f)

    ans = ans1(X_train, y_train)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
