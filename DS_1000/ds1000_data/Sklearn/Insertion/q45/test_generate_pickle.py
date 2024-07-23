import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def define_test_input(args):
    if args.test_case == 1:
        X, y = make_blobs(n_samples=200, n_features=3, centers=8, random_state=42)
        p = 2
    elif args.test_case == 2:
        X, y = make_blobs(n_samples=200, n_features=3, centers=8, random_state=42)
        p = 3
    return p, X


def ans1(p, X):
    km = KMeans(n_clusters=8, random_state=42)
    km.fit(X)
    d = km.transform(X)[:, p]
    indexes = np.argsort(d)[::][:50]
    closest_50_samples = X[indexes]
    return closest_50_samples


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    p, X = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((p, X), f)

    ans = ans1(p, X)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
