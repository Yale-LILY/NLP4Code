import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.svm import SVR


def define_test_input(args):
    if args.test_case == 1:
        X, y = make_regression(n_samples=1000, n_features=4, random_state=42)
    return X, y


def ans1(X, y):
    svr_poly = SVR(kernel="poly", degree=2)
    svr_poly.fit(X, y)
    predict = svr_poly.predict(X)
    return predict


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
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
