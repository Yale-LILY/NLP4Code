import numpy as np
import pandas as pd
import scipy
from scipy import sparse

from scipy import stats
import pandas as pd
import numpy as np


def define_test_input(args):
    if args.test_case == 1:
        LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        np.random.seed(17)
        df = pd.DataFrame(
            {
                "NUM1": np.random.randn(50) * 100,
                "NUM2": np.random.uniform(0, 1, 50),
                "NUM3": np.random.randint(100, size=50),
                "CAT1": ["".join(np.random.choice(LETTERS, 1)) for _ in range(50)],
                "CAT2": [
                    "".join(
                        np.random.choice(
                            ["pandas", "r", "julia", "sas", "stata", "spss"], 1
                        )
                    )
                    for _ in range(50)
                ],
                "CAT3": [
                    "".join(
                        np.random.choice(
                            [
                                "postgres",
                                "mysql",
                                "sqlite",
                                "oracle",
                                "sql server",
                                "db2",
                            ],
                            1,
                        )
                    )
                    for _ in range(50)
                ],
            }
        )
    return df


def ans1(_a):
    df = _a
    df = df[(np.abs(stats.zscore(df.select_dtypes(exclude="object"))) < 3).all(axis=1)]

    return df


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
