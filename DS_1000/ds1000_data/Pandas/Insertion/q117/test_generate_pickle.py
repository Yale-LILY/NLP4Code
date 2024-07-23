import pandas as pd
import numpy as np
import os
import io

# test function
def g(df, test):
    return df.loc[test]


def define_test_input(args):
    if args.test_case == 1:
        data = io.StringIO(
            """
        rs  alleles  chrom  pos strand  assembly#  center  protLSID  assayLSID
        TP3      A/C      0    3      +        NaN     NaN       NaN        NaN
        TP7      A/T      0    7      +        NaN     NaN       NaN        NaN
        TP12     T/A      0   12      +        NaN     NaN       NaN        NaN
        TP15     C/A      0   15      +        NaN     NaN       NaN        NaN
        TP18     C/T      0   18      +        NaN     NaN       NaN        NaN
        """
        )
        df = pd.read_csv(data, delim_whitespace=True).set_index("rs")
        test = ["TP3", "TP7", "TP18"]

    return df, test


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df, test = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df, test), f)

    result = g(df, test)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
