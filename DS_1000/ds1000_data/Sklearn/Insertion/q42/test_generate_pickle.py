import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def define_test_input(args):
    if args.test_case == 1:
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X = pd.DataFrame(
            X,
            columns=[
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            ],
        )
        y = pd.Series(y)
    return X, y


def ans1(X, y):
    clf = ExtraTreesClassifier(random_state=42)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    column_names = X.columns[model.get_support()]
    return column_names


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
    # pd.testing.assert_index_equal(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
