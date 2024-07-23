import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(42)
        GridSearch_fitted = GridSearchCV(LogisticRegression(), {"C": [1, 2, 3]})
        GridSearch_fitted.fit(np.random.randn(50, 4), np.random.randint(0, 2, 50))
    return GridSearch_fitted


def ans1(GridSearch_fitted):
    full_results = pd.DataFrame(GridSearch_fitted.cv_results_).sort_values(
        by="mean_fit_time", ascending=True
    )
    return full_results


def ans2(GridSearch_fitted):
    full_results = pd.DataFrame(GridSearch_fitted.cv_results_).sort_values(
        by="mean_fit_time", ascending=False
    )
    return full_results


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    GridSearch_fitted = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(GridSearch_fitted, f)

    ans = ans1(GridSearch_fitted), ans2(GridSearch_fitted)
    # print(ans)
    # np.testing.assert_equal(ans, result)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
