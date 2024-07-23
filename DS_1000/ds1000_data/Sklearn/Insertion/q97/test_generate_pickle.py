import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def define_test_input(args):
    if args.test_case == 1:
        X = np.array([[-1, 2], [-0.5, 6]])
    return X


def ans1(X):
    scaler = MinMaxScaler()
    X_one_column = X.reshape([-1, 1])
    result_one_column = scaler.fit_transform(X_one_column)
    result = result_one_column.reshape(X.shape)
    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    X = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(X, f)

    ans = ans1(X)
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
