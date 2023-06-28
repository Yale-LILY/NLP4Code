import pandas as pd
import numpy as np


def g(df):
    df["cumsum"] = df.groupby("id")["val"].transform(pd.Series.cumsum)
    df["cumsum"] = df["cumsum"].where(df["cumsum"] > 0, 0)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame.from_dict(
            {
                "id": ["A", "B", "A", "C", "D", "B", "C"],
                "val": [1, 2, -3, 1, 5, 6, -2],
                "stuff": ["12", "23232", "13", "1234", "3235", "3236", "732323"],
            }
        )
    elif args.test_case == 2:
        np.random.seed(19260817)
        df = pd.DataFrame(
            {"id": ["A", "B"] * 10 + ["C"] * 10, "val": np.random.randint(0, 100, 30)}
        )
    elif args.test_case == 3:
        np.random.seed(19260817)
        df = pd.DataFrame(
            {
                "id": np.random.choice(list("ABCDE"), 1000),
                "val": np.random.randint(-1000, 1000, 1000),
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
