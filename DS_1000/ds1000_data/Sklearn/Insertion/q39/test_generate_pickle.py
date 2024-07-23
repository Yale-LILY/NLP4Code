import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def define_test_input(args):
    if args.test_case == 1:
        X, y = make_classification(random_state=42)
    return X, y


def ans1(X, y):
    pipe = Pipeline(
        [("scale", StandardScaler()), ("model", SGDClassifier(random_state=42))]
    )
    grid = GridSearchCV(pipe, param_grid={"model__alpha": [1e-3, 1e-2, 1e-1, 1]}, cv=5)
    grid.fit(X, y)
    coef = grid.best_estimator_.named_steps["model"].coef_
    return coef


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

    ans = ans1(X, y)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
