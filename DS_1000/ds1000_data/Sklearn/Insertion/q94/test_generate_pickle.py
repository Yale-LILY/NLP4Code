import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def define_test_input(args):
    if args.test_case == 1:
        X_train, y_train = make_regression(
            n_samples=1000, n_features=5, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.4, random_state=42
        )
    return X_train, y_train, X_test, y_test


def ans1(X_train, y_train, X_test, y_test):
    ElasticNet = linear_model.ElasticNet()
    ElasticNet.fit(X_train, y_train)
    training_set_score = ElasticNet.score(X_train, y_train)
    test_set_score = ElasticNet.score(X_test, y_test)
    return training_set_score, test_set_score


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)

    ans = ans1(X_train, y_train, X_test, y_test)
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
