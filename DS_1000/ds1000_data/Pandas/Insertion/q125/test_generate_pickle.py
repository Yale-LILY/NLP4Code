import pandas as pd
import numpy as np

import numpy as np


def g(df):
    df["keywords_all"] = df.filter(like="keyword").apply(
        lambda x: "-".join(x.dropna()), axis=1
    )
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "users": ["Hu Tao", "Zhongli", "Xingqiu"],
                "keywords_0": ["a", np.nan, "c"],
                "keywords_1": ["d", "e", np.nan],
                "keywords_2": [np.nan, np.nan, "b"],
                "keywords_3": ["f", np.nan, "g"],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "users": ["Hu Tao", "Zhongli", "Xingqiu"],
                "keywords_0": ["a", np.nan, "c"],
                "keywords_1": ["b", "g", np.nan],
                "keywords_2": [np.nan, np.nan, "j"],
                "keywords_3": ["d", np.nan, "b"],
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
