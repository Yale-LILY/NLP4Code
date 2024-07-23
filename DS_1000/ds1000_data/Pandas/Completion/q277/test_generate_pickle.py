import pandas as pd
import numpy as np


def g(df):
    return df.sort_values("VIM")


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "VIM": [
                    -0.158406,
                    0.039158,
                    -0.052608,
                    0.157153,
                    0.206030,
                    0.132580,
                    -0.144209,
                    -0.093910,
                    -0.166819,
                    0.097548,
                    0.026664,
                    -0.008032,
                ]
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("TGFb", 0.1, 2),
                    ("TGFb", 1, 2),
                    ("TGFb", 10, 2),
                    ("TGFb", 0.1, 24),
                    ("TGFb", 1, 24),
                    ("TGFb", 10, 24),
                    ("TGFb", 0.1, 48),
                    ("TGFb", 1, 48),
                    ("TGFb", 10, 48),
                    ("TGFb", 0.1, 6),
                    ("TGFb", 1, 6),
                    ("TGFb", 10, 6),
                ],
                names=["treatment", "dose", "time"],
            ),
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
