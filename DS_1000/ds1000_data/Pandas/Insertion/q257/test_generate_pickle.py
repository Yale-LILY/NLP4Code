import pandas as pd
import numpy as np


def g(df):
    df.set_index("Time", inplace=True)
    df_group = df.groupby(pd.Grouper(level="Time", freq="2T"))["Value"].agg("mean")
    df_group.dropna(inplace=True)
    df_group = df_group.to_frame().reset_index()
    return df_group


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Time": [
                    "2015-04-24 06:38:49",
                    "2015-04-24 06:39:19",
                    "2015-04-24 06:43:49",
                    "2015-04-24 06:44:18",
                    "2015-04-24 06:44:48",
                    "2015-04-24 06:45:18",
                    "2015-04-24 06:47:48",
                    "2015-04-24 06:48:18",
                    "2015-04-24 06:50:48",
                    "2015-04-24 06:51:18",
                    "2015-04-24 06:51:48",
                    "2015-04-24 06:52:18",
                    "2015-04-24 06:52:48",
                    "2015-04-24 06:53:48",
                    "2015-04-24 06:55:18",
                    "2015-04-24 07:00:47",
                    "2015-04-24 07:01:17",
                    "2015-04-24 07:01:47",
                ],
                "Value": [
                    0.023844,
                    0.019075,
                    0.023844,
                    0.019075,
                    0.023844,
                    0.019075,
                    0.023844,
                    0.019075,
                    0.023844,
                    0.019075,
                    0.023844,
                    0.019075,
                    0.023844,
                    0.019075,
                    0.023844,
                    0.019075,
                    0.023844,
                    0.019075,
                ],
            }
        )
        df["Time"] = pd.to_datetime(df["Time"])
    if args.test_case == 2:
        np.random.seed(4)
        df = pd.DataFrame(
            {
                "Time": [
                    "2015-04-24 06:38:49",
                    "2015-04-24 06:39:19",
                    "2015-04-24 06:43:49",
                    "2015-04-24 06:44:18",
                    "2015-04-24 06:44:48",
                    "2015-04-24 06:45:18",
                    "2015-04-24 06:47:48",
                    "2015-04-24 06:48:18",
                    "2015-04-24 06:50:48",
                    "2015-04-24 06:51:18",
                    "2015-04-24 06:51:48",
                    "2015-04-24 06:52:18",
                    "2015-04-24 06:52:48",
                    "2015-04-24 06:53:48",
                    "2015-04-24 06:55:18",
                    "2015-04-24 07:00:47",
                    "2015-04-24 07:01:17",
                    "2015-04-24 07:01:47",
                ],
                "Value": np.random.random(18),
            }
        )
        df["Time"] = pd.to_datetime(df["Time"])
    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    result = g(df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
