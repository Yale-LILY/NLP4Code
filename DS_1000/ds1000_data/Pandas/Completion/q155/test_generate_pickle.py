import pandas as pd
import numpy as np


def g(df):
    uniq_indx = (
        df.sort_values(by="bank", na_position="last")
        .dropna(subset=["firstname", "lastname", "email"])
        .applymap(lambda s: s.lower() if type(s) == str else s)
        .applymap(lambda x: x.replace(" ", "") if type(x) == str else x)
        .drop_duplicates(subset=["firstname", "lastname", "email"], keep="first")
    ).index
    return df.loc[uniq_indx]


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "firstname": ["foo Bar", "Bar Bar", "Foo Bar"],
                "lastname": ["Foo Bar", "Bar", "Foo Bar"],
                "email": ["Foo bar", "Bar", "Foo Bar"],
                "bank": [np.nan, "abc", "xyz"],
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
