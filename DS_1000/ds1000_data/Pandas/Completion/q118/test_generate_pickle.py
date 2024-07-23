import pandas as pd
import numpy as np

# test function
def g(df, test):
    return df.loc[test]


def define_test_input(args):
    if args.test_case == 1:
        import io

        data = io.StringIO(
            """
        rs    alias  chrome  poston 
        TP3      A/C      0    3  
        TP7      A/T      0    7
        TP12     T/A      0   12 
        TP15     C/A      0   15
        TP18     C/T      0   18 
        """
        )
        df = pd.read_csv(data, delim_whitespace=True).set_index("rs")
        test = ["TP3", "TP18"]
    return df, test


if __name__ == "__main__":
    import argparse
    import os
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
