import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Col1": ["C", "A", "B"],
                "Col2": ["33", "2.5", "42"],
                "Col3": [["Apple", "Orange", "Banana"], ["Apple", "Grape"], ["Banana"]],
            }
        )
    elif args.test_case == 2:
        df = pd.DataFrame(
            {
                "Col1": ["c", "a", "b"],
                "Col2": ["3", "2", "4"],
                "Col3": [["Apple", "Orange", "Banana"], ["Apple", "Grape"], ["Banana"]],
            }
        )
    elif args.test_case == 3:
        df = pd.DataFrame(
            {
                "Col1": ["c", "a", "b"],
                "Col2": ["3", "2", "4"],
                "Col3": [
                    ["Apple", "Orange", "Banana", "Watermelon"],
                    ["Apple", "Grape"],
                    ["Banana"],
                ],
            }
        )
    elif args.test_case == 4:
        df = pd.DataFrame(
            {
                "Col1": ["C", "A", "B", "D"],
                "Col2": ["33", "2.5", "42", "666"],
                "Col3": ["11", "4.5", "14", "1919810"],
                "Col4": [
                    ["Apple", "Orange", "Banana"],
                    ["Apple", "Grape"],
                    ["Banana"],
                    ["Suica", "Orange"],
                ],
            }
        )
    return df


def ans1(df):
    mlb = MultiLabelBinarizer()

    df_out = df.join(
        pd.DataFrame(
            mlb.fit_transform(df.pop(df.columns[-1])),
            index=df.index,
            columns=mlb.classes_,
        )
    )
    return df_out


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df = define_test_input(args)
    # print(df)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    ans = ans1(df)
    # print(ans)
    # np.testing.assert_equal(ans, ans)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
