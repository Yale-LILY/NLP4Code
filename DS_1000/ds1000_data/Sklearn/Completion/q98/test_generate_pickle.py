import numpy as np
import pandas as pd
from sklearn import tree


def define_test_input(args):
    if args.test_case == 1:
        with open("DataFrame.pkl", "rb") as f:
            df = pickle.load(f)
    return df


def ans1(df):
    # df = web.DataReader('goog', 'yahoo', start='2012-5-1', end='2016-5-20')

    df["B/S"] = (df["Close"].diff() < 0).astype(int)

    closing = df.loc["2013-02-15":"2016-05-21"]
    ma_50 = df.loc["2013-02-15":"2016-05-21"]
    ma_100 = df.loc["2013-02-15":"2016-05-21"]
    ma_200 = df.loc["2013-02-15":"2016-05-21"]
    buy_sell = df.loc["2013-02-15":"2016-05-21"]  # Fixed

    close = pd.DataFrame(closing)
    ma50 = pd.DataFrame(ma_50)
    ma100 = pd.DataFrame(ma_100)
    ma200 = pd.DataFrame(ma_200)
    buy_sell = pd.DataFrame(buy_sell)

    clf = tree.DecisionTreeRegressor(random_state=42)
    x = np.concatenate([close, ma50, ma100, ma200], axis=1)
    y = buy_sell

    clf.fit(x, y)
    close_buy1 = close[:-1]
    m5 = ma_50[:-1]
    m10 = ma_100[:-1]
    ma20 = ma_200[:-1]
    # b = np.concatenate([close_buy1, m5, m10, ma20], axis=1)

    predict = clf.predict(pd.concat([close_buy1, m5, m10, ma20], axis=1))
    return predict


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
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
