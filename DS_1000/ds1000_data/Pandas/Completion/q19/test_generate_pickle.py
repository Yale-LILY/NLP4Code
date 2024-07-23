import pandas as pd
import numpy as np


def g(df, prod_list):
    Max = df.loc[df["product"].isin(prod_list), "score"].max()
    Min = df.loc[df["product"].isin(prod_list), "score"].min()
    df.loc[df["product"].isin(prod_list), "score"] = (
        df.loc[df["product"].isin(prod_list), "score"] - Min
    ) / (Max - Min)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "product": [
                    1179160,
                    1066490,
                    1148126,
                    1069104,
                    1069105,
                    1160330,
                    1069098,
                    1077784,
                    1193369,
                    1179741,
                ],
                "score": [
                    0.424654,
                    0.424509,
                    0.422207,
                    0.420455,
                    0.414603,
                    0.168784,
                    0.168749,
                    0.168738,
                    0.168703,
                    0.168684,
                ],
            }
        )
        products = [1066490, 1077784, 1179741]

    return df, products


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, prod_list = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, prod_list), f)

    result = g(df, prod_list)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
