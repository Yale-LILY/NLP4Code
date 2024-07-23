import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# def define_test_input(args):
#     if args.test_case == 1:
#         df = pd.DataFrame({'date': ['2017-03-01', '2017-03-02', '2017-03-03', '2017-03-04', '2017-03-05', '2017-03-06',
#                                     '2017-03-07', '2017-03-08', '2017-03-09', '2017-03-10'],
#                            'sales': [12000, 8000, 25000, 15000, 10000, 15000, 10000, 25000, 12000, 15000],
#                            'profit': [18000, 12000, 30000, 20000, 15000, 20000, 15000, 30000, 18000, 20000]
#                            })
#     elif args.test_case == 2:
#         df = pd.DataFrame({'date': [datetime.datetime(2020, 7, 1), datetime.datetime(2020, 7, 2),
#                                     datetime.datetime(2020, 7, 3), datetime.datetime(2020, 7, 4),
#                                     datetime.datetime(2020, 7, 5), datetime.datetime(2020, 7, 6),
#                                     datetime.datetime(2020, 7, 7), datetime.datetime(2020, 7, 8),
#                                     datetime.datetime(2020, 7, 9), datetime.datetime(2020, 7, 10),
#                                     datetime.datetime(2020, 7, 11), datetime.datetime(2020, 7, 12),
#                                     datetime.datetime(2020, 7, 13), datetime.datetime(2020, 7, 14),
#                                     datetime.datetime(2020, 7, 15), datetime.datetime(2020, 7, 16),
#                                     datetime.datetime(2020, 7, 17), datetime.datetime(2020, 7, 18),
#                                     datetime.datetime(2020, 7, 19), datetime.datetime(2020, 7, 20),
#                                     datetime.datetime(2020, 7, 21), datetime.datetime(2020, 7, 22),
#                                     datetime.datetime(2020, 7, 23), datetime.datetime(2020, 7, 24),
#                                     datetime.datetime(2020, 7, 25), datetime.datetime(2020, 7, 26),
#                                     datetime.datetime(2020, 7, 27), datetime.datetime(2020, 7, 28),
#                                     datetime.datetime(2020, 7, 29), datetime.datetime(2020, 7, 30),
#                                     datetime.datetime(2020, 7, 31)],
#                            'counts': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
#                                       24, 25, 26, 27, 28, 29, 30, 31]})
#     return df


def ans1():
    myData = pd.DataFrame(
        {
            "Month": [3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
            "A1": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            "A2": [31, 13, 13, 13, 33, 33, 81, 38, 18, 38, 18, 18, 118],
            "A3": [81, 38, 18, 38, 18, 18, 118, 31, 13, 13, 13, 33, 33],
            "A4": [1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8],
        }
    )
    scaler = MinMaxScaler()
    cols = myData.columns[2:4]

    def scale(X):
        X_ = np.atleast_2d(X)
        return pd.DataFrame(scaler.fit_transform(X_), X.index)

    myData["new_" + cols] = myData.groupby("Month")[cols].apply(scale)
    return myData


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    # features_dataframe = define_test_input(args)
    #
    # if not os.path.exists("input"):
    #     os.mkdir("input")
    # with open('input/input{}.pkl'.format(args.test_case), 'wb') as f:
    #     pickle.dump(features_dataframe, f)

    ans = ans1()
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # print(ans.toarray())
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
