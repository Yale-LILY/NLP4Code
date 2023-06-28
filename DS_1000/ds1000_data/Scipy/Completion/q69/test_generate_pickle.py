import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import io


import pandas as pd
import io
import numpy as np
from scipy import stats


def define_test_input(args):
    if args.test_case == 1:
        temp = """probegenes,sample1,sample2,sample3
1415777_at Pnliprp1,20,0.00,11
1415805_at Clps,17,0.00,55
1415884_at Cela3b,47,0.00,100"""

    elif args.test_case == 2:
        temp = """probegenes,sample1,sample2,sample3
1415777_at Pnliprp1,20,2.00,11
1415805_at Clps,17,0.30,55
1415884_at Cela3b,47,1.00,100"""
    df = pd.read_csv(io.StringIO(temp), index_col="probegenes")

    return df


def ans1(_a):
    df = _a
    indices = [
        ("1415777_at Pnliprp1", "data"),
        ("1415777_at Pnliprp1", "zscore"),
        ("1415805_at Clps", "data"),
        ("1415805_at Clps", "zscore"),
        ("1415884_at Cela3b", "data"),
        ("1415884_at Cela3b", "zscore"),
    ]
    indices = pd.MultiIndex.from_tuples(indices)
    df2 = pd.DataFrame(
        data=stats.zscore(df, axis=0), index=df.index, columns=df.columns
    )
    df3 = pd.concat([df, df2], axis=1).to_numpy().reshape(-1, 3)
    result = pd.DataFrame(data=np.round(df3, 3), index=indices, columns=df.columns)

    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(a)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
