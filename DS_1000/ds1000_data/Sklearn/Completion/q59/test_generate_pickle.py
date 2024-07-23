import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def define_test_input(args):
    if args.test_case == 1:
        f = [["f1", "f2", "f3"], ["f2", "f4", "f5", "f6"], ["f1", "f2"]]
    elif args.test_case == 2:
        f = [
            ["t1"],
            ["t2", "t5", "t7"],
            ["t1", "t2", "t3", "t4", "t5"],
            ["t4", "t5", "t6"],
        ]
    return f


def ans1(f):
    from sklearn.preprocessing import MultiLabelBinarizer

    new_f = MultiLabelBinarizer().fit_transform(f)
    return new_f


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    f = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as _f:
        pickle.dump(f, _f)

    ans = ans1(f)
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as _f:
        pickle.dump(ans, _f)
