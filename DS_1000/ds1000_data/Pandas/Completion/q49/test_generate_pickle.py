import pandas as pd
import numpy as np


def g(df, section_left, section_right):
    return df[lambda x: x["value"].between(section_left, section_right)].append(
        df[lambda x: ~x["value"].between(section_left, section_right)]
        .mean()
        .rename("X")
    )


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {"lab": ["A", "B", "C", "D", "E", "F"], "value": [50, 35, 8, 5, 1, 1]}
        )
        df = df.set_index("lab")
        section_left = 4
        section_right = 38
    if args.test_case == 2:
        df = pd.DataFrame(
            {"lab": ["A", "B", "C", "D", "E", "F"], "value": [50, 35, 8, 5, 1, 1]}
        )
        df = df.set_index("lab")
        section_left = 6
        section_right = 38
    return df, section_left, section_right


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, section_left, section_right = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, section_left, section_right), f)

    result = g(df, section_left, section_right)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
