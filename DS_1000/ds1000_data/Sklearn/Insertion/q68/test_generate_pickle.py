import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree


def define_test_input(args):
    if args.test_case == 1:
        data_matrix = [[0, 0.8, 0.9], [0.8, 0, 0.2], [0.9, 0.2, 0]]
    elif args.test_case == 2:
        data_matrix = [[0, 0.2, 0.9], [0.2, 0, 0.8], [0.9, 0.8, 0]]
    return data_matrix


def ans1(simM):
    Z = linkage(np.array(simM), "ward")
    cluster_labels = (
        cut_tree(Z, n_clusters=2)
        .reshape(
            -1,
        )
        .tolist()
    )
    return cluster_labels


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    simM = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(simM, f)

    ans = ans1(simM)
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
