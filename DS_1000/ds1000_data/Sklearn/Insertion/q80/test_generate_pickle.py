import numpy as np
import pandas as pd
import pandas.testing
from sklearn.cluster import KMeans


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "date": [
                    "2018-02-11",
                    "2018-02-12",
                    "2018-02-13",
                    "2018-02-14",
                    "2018-02-16",
                    "2018-02-21",
                    "2018-02-22",
                    "2018-02-24",
                    "2018-02-26",
                    "2018-02-27",
                    "2018-02-28",
                    "2018-03-01",
                    "2018-03-05",
                    "2018-03-06",
                ],
                "mse": [
                    14.34,
                    7.24,
                    4.5,
                    3.5,
                    12.67,
                    45.66,
                    15.33,
                    98.44,
                    23.55,
                    45.12,
                    78.44,
                    34.11,
                    23.33,
                    7.45,
                ],
            }
        )
    return df


def ans1(df):
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(df[["mse"]])
    return labels


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    ans = ans1(df)
    # print(1 - ans)
    # kmeans = KMeans(n_clusters=2)
    # labels = kmeans.fit_predict(df[['mse']])
    # np.testing.assert_equal(labels, ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
