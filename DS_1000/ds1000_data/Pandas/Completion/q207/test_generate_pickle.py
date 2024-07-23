import pandas as pd
import numpy as np


def g(df):
    label = [
        1,
    ]
    for i in range(1, len(df)):
        if df.loc[i, "Close"] > df.loc[i - 1, "Close"]:
            label.append(1)
        elif df.loc[i, "Close"] == df.loc[i - 1, "Close"]:
            label.append(0)
        else:
            label.append(-1)
    df["label"] = label
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "DateTime": [
                    "2000-01-04",
                    "2000-01-05",
                    "2000-01-06",
                    "2000-01-07",
                    "2000-01-08",
                ],
                "Close": [1460, 1470, 1480, 1480, 1450],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "DateTime": [
                    "2000-02-04",
                    "2000-02-05",
                    "2000-02-06",
                    "2000-02-07",
                    "2000-02-08",
                ],
                "Close": [1460, 1470, 1480, 1480, 1450],
            }
        )
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
