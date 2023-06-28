import numpy as np
import pandas as pd
from sklearn import datasets


def define_test_input(args):
    if args.test_case == 1:
        iris = datasets.load_iris()
        X = iris.data[(iris.target == 0) | (iris.target == 1)]
        Y = iris.target[(iris.target == 0) | (iris.target == 1)]

        # Class 0 has indices 0-49. Class 1 has indices 50-99.
        # Divide data into 80% training, 20% testing.
        train_indices = list(range(40)) + list(range(50, 90))
        test_indices = list(range(40, 50)) + list(range(90, 100))
        X_train = X[train_indices]
        y_train = Y[train_indices]

        X_train = pd.DataFrame(X_train)
    return X_train, y_train


def ans1(X_train, y_train):
    X_train[0] = ["a"] * 40 + ["b"] * 40
    ### BEGIN SOLUTION
    catVar = pd.get_dummies(X_train[0]).to_numpy()
    X_train = np.concatenate((X_train.iloc[:, 1:], catVar), axis=1)
    return X_train


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
    # # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
