import pandas as pd
import numpy as np


def g(s):
    return pd.to_numeric(s.str.replace(",", ""), errors="coerce")


def define_test_input(args):
    if args.test_case == 1:
        s = pd.Series(
            [
                "2,144.78",
                "2,036.62",
                "1,916.60",
                "1,809.40",
                "1,711.97",
                "6,667.22",
                "5,373.59",
                "4,071.00",
                "3,050.20",
                "-0.06",
                "-1.88",
                "",
                "-0.13",
                "",
                "-0.14",
                "0.07",
                "0",
                "0",
            ],
            index=[
                "2016-10-31",
                "2016-07-31",
                "2016-04-30",
                "2016-01-31",
                "2015-10-31",
                "2016-01-31",
                "2015-01-31",
                "2014-01-31",
                "2013-01-31",
                "2016-09-30",
                "2016-06-30",
                "2016-03-31",
                "2015-12-31",
                "2015-09-30",
                "2015-12-31",
                "2014-12-31",
                "2013-12-31",
                "2012-12-31",
            ],
        )
    if args.test_case == 2:
        s = pd.Series(
            [
                "2,144.78",
                "2,036.62",
                "1,916.60",
                "1,809.40",
                "1,711.97",
                "6,667.22",
                "5,373.59",
                "4,071.00",
                "3,050.20",
                "-0.06",
                "-1.88",
                "",
                "-0.13",
                "",
                "-0.14",
                "0.07",
                "0",
                "0",
            ],
            index=[
                "2026-10-31",
                "2026-07-31",
                "2026-04-30",
                "2026-01-31",
                "2025-10-31",
                "2026-01-31",
                "2025-01-31",
                "2024-01-31",
                "2023-01-31",
                "2026-09-30",
                "2026-06-30",
                "2026-03-31",
                "2025-12-31",
                "2025-09-30",
                "2025-12-31",
                "2024-12-31",
                "2023-12-31",
                "2022-12-31",
            ],
        )
    return s


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    s = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(s, f)

    result = g(s)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
