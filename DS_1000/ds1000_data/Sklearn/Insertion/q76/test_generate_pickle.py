import numpy as np
import pandas as pd
import pandas.testing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def define_test_input(args):
    if args.test_case == 1:
        data, target = load_iris(as_frame=True, return_X_y=True)
        dataset = pd.concat([data, target], axis=1)
    return dataset


def ans1(dataset):
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    dataset = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(dataset, f)

    ans = ans1(dataset)
    # ans = ans1(data.copy())
    # result = ans1(data.copy())
    # print(ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
