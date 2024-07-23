import pandas as pd
import numpy as np


def justify(a, invalid_val=0, axis=1, side="left"):
    if invalid_val is np.nan:
        mask = ~np.isnan(a)
    else:
        mask = a != invalid_val
    justified_mask = np.sort(mask, axis=axis)
    if (side == "up") | (side == "left"):
        justified_mask = np.flip(justified_mask, axis=axis)
    out = np.full(a.shape, invalid_val)
    if axis == 1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out


def g(df):
    return pd.DataFrame(justify(df.values, invalid_val=np.nan, axis=1, side="right"))


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            [[3, 1, 2], [1, 2, np.nan], [2, np.nan, np.nan]], columns=["0", "1", "2"]
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
