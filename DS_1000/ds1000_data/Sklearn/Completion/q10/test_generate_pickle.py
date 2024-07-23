import numpy as np
import pandas as pd
import sklearn.svm as suppmach
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification, load_iris


def define_test_input(args):
    if args.test_case == 1:
        X, y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, random_state=42
        )
        x_predict = X
    elif args.test_case == 2:
        X, y = load_iris(return_X_y=True)
        x_predict = X
    return X, y, x_predict


def ans1(X, y, x_predict):
    svmmodel = suppmach.LinearSVC(random_state=42)
    calibrated_svc = CalibratedClassifierCV(svmmodel, cv=5, method="sigmoid")
    calibrated_svc.fit(X, y)
    proba = calibrated_svc.predict_proba(x_predict)
    return proba


def ans2(X, y, x_predict):
    svmmodel = suppmach.LinearSVC(random_state=42)
    calibrated_svc = CalibratedClassifierCV(svmmodel, cv=5, method="isotonic")
    calibrated_svc.fit(X, y)
    proba = calibrated_svc.predict_proba(x_predict)
    return proba


def ans3(X, y, x_predict):
    svmmodel = suppmach.LinearSVC(random_state=42)
    calibrated_svc = CalibratedClassifierCV(
        svmmodel, cv=5, method="sigmoid", ensemble=False
    )
    calibrated_svc.fit(X, y)
    proba = calibrated_svc.predict_proba(x_predict)
    return proba


def ans4(X, y, x_predict):
    svmmodel = suppmach.LinearSVC(random_state=42)
    calibrated_svc = CalibratedClassifierCV(
        svmmodel, cv=5, method="isotonic", ensemble=False
    )
    calibrated_svc.fit(X, y)
    proba = calibrated_svc.predict_proba(x_predict)
    return proba


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    X, y, x_predict = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((X, y, x_predict), f)

    ans = (
        ans1(X.copy(), y.copy(), x_predict.copy()),
        ans2(X.copy(), y.copy(), x_predict.copy()),
        ans3(X.copy(), y.copy(), x_predict.copy()),
        ans4(X.copy(), y.copy(), x_predict.copy()),
    )
    # np.testing.assert_equal(ans[0], ans[1])
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
