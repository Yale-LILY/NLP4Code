import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor


def define_test_input(args):
    if args.test_case == 1:
        X, y = make_regression(
            n_samples=100,
            n_features=1,
            n_informative=1,
            bias=150.0,
            noise=30,
            random_state=42,
        )
    return X.flatten(), y, X


def ans1(X, y, X_test):
    regressor = RandomForestRegressor(
        n_estimators=150, min_samples_split=1.0, random_state=42
    )
    regressor.fit(X.reshape(-1, 1), y)
    predict = regressor.predict(X_test)
    return predict


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    X, y, X_test = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((X, y, X_test), f)

    ans = ans1(X, y, X_test)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
