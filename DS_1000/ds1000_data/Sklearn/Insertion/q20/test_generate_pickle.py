import numpy as np
import pandas as pd
import sklearn
import xgboost.sklearn as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


def define_test_input(args):
    if args.test_case == 1:
        trainX = [[1], [2], [3], [4], [5]]
        trainY = [1, 2, 3, 4, 5]
        testX, testY = trainX, trainY
        paramGrid = {"subsample": [0.5, 0.8]}
        model = xgb.XGBRegressor()
        gridsearch = GridSearchCV(
            model,
            paramGrid,
            cv=TimeSeriesSplit(n_splits=2).get_n_splits([trainX, trainY]),
        )
    return gridsearch, testX, testY, trainX, trainY


def ans1(gridsearch, testX, testY, trainX, trainY):
    fit_params = {
        "early_stopping_rounds": 42,
        "eval_metric": "mae",
        "eval_set": [[testX, testY]],
    }
    gridsearch.fit(trainX, trainY, **fit_params)
    b = gridsearch.score(trainX, trainY)
    c = gridsearch.predict(trainX)
    return b, c


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    gridsearch, testX, testY, trainX, trainY = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((gridsearch, testX, testY, trainX, trainY), f)

    ans = ans1(gridsearch, testX, testY, trainX, trainY)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
