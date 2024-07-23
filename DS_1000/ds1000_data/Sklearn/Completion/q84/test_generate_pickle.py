import numpy as np
import pandas as pd
import pandas.testing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def define_test_input(args):
    if args.test_case == 1:
        corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the first piece of news",
            "Is this the first document? No, it'the fourth document",
            "This is the second news",
        ]
        y = [0, 0, 1, 0, 1]
    return corpus, y


def ans1(corpus, y):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    svc = LinearSVC(penalty="l1", dual=False)
    svc.fit(X, y)
    selected_feature_names = np.asarray(vectorizer.get_feature_names_out())[
        np.flatnonzero(svc.coef_)
    ]
    return selected_feature_names


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    corpus, y = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((corpus, y), f)

    ans = ans1(corpus, y)
    # print(ans)
    # np.testing.assert_equal(ans, ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
