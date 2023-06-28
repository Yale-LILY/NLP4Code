import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def define_test_input(args):
    if args.test_case == 1:
        df1 = pd.DataFrame(
            {
                "Time": [1, 2, 3, 4, 5, 5.5, 6],
                "A1": [6.64, 6.70, None, 7.15, None, 7.44, 7.62],
                "A2": [6.82, 6.86, None, 7.26, None, 7.63, 7.86],
                "A3": [6.79, 6.92, None, 7.26, None, 7.58, 7.71],
                "B1": [6.70, None, 7.07, 7.19, None, 7.54, None],
                "B2": [6.95, None, 7.27, None, 7.40, None, None],
                "B3": [7.02, None, 7.40, None, 7.51, None, None],
            }
        )
    elif args.test_case == 2:
        df1 = pd.DataFrame(
            {
                "Time": [1, 2, 3, 4, 5, 5.5],
                "A1": [6.64, 6.70, np.nan, 7.15, np.nan, 7.44],
                "A2": [6.82, 6.86, np.nan, 7.26, np.nan, 7.63],
                "A3": [6.79, 6.92, np.nan, 7.26, np.nan, 7.58],
                "B1": [6.70, np.nan, 7.07, 7.19, np.nan, 7.54],
                "B2": [6.95, np.nan, 7.27, np.nan, 7.40, np.nan],
                "B3": [7.02, np.nan, 7.40, 6.95, 7.51, 6.95],
                "C1": [np.nan, 6.95, np.nan, 7.02, np.nan, 7.02],
                "C2": [np.nan, 7.02, np.nan, np.nan, 6.95, np.nan],
                "C3": [6.95, 6.95, 6.95, 6.95, 7.02, 6.95],
                "D1": [7.02, 7.02, 7.02, 7.02, np.nan, 7.02],
                "D2": [np.nan, 3.14, np.nan, 9.28, np.nan, np.nan],
                "D3": [6.95, 6.95, 6.95, 6.95, 6.95, 6.95],
            }
        )
    return df1


def ans1(df1):
    slopes = []
    for col in df1.columns:
        if col == "Time":
            continue
        mask = ~np.isnan(df1[col])
        x = np.atleast_2d(df1.Time[mask].values).T
        y = np.atleast_2d(df1[col][mask].values).T
        reg = LinearRegression().fit(x, y)
        slopes.append(reg.coef_[0])
    slopes = np.array(slopes).reshape(-1)
    return slopes


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df1 = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df1, f)

    ans = ans1(df1)
    # np.testing.assert_allclose(ans, ans)
    # print(ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
