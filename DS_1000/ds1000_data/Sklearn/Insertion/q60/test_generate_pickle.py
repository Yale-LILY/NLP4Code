import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def define_test_input(args):
    if args.test_case == 1:
        features = [["f1", "f2", "f3"], ["f2", "f4", "f5", "f6"], ["f1", "f2"]]
    return features


def ans1(features):
    from sklearn.preprocessing import MultiLabelBinarizer

    new_features = MultiLabelBinarizer().fit_transform(features)
    rows, cols = new_features.shape
    for i in range(rows):
        for j in range(cols):
            if new_features[i, j] == 1:
                new_features[i, j] = 0
            else:
                new_features[i, j] = 1
    return new_features


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    features = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(features, f)

    ans = ans1(features)
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
