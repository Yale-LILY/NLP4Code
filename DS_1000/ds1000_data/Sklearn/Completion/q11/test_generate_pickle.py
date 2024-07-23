import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def define_test_input(args):
    if args.test_case == 1:
        df_origin = pd.DataFrame([[1, 1, 4], [0, 3, 0]], columns=["A", "B", "C"])
        transform_output = csr_matrix([[1, 0, 2], [0, 3, 0]])
    elif args.test_case == 2:
        df_origin = pd.DataFrame(
            [[1, 1, 4, 5], [1, 4, 1, 9]], columns=["A", "B", "C", "D"]
        )
        transform_output = csr_matrix([[1, 9, 8, 1, 0], [1, 1, 4, 5, 1]])
    return df_origin, transform_output


def ans1(df_origin, transform_output):
    df = pd.concat([df_origin, pd.DataFrame(transform_output.toarray())], axis=1)
    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df_origin, transform_output = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((df_origin, transform_output), f)

    ans = ans1(df_origin, transform_output)
    # print(ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
