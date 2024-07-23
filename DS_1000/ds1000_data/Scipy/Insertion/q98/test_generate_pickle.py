import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import numpy as np
import scipy as sp
from scipy import integrate, stats


def define_test_input(args):
    range_start = 1
    range_end = 10
    estimated_a, estimated_m, estimated_d = 1, 1, 1
    if args.test_case == 1:
        sample_data = [1.5, 1.6, 1.8, 2.1, 2.2, 3.3, 4, 6, 8, 9]
    elif args.test_case == 2:
        sample_data = [1, 1, 1, 1, 1, 3.3, 4, 6, 8, 9]
    return range_start, range_end, estimated_a, estimated_m, estimated_d, sample_data


def ans1(_a):
    def bekkers(x, a, m, d):
        p = a * np.exp((-1 * (x ** (1 / 3) - m) ** 2) / (2 * d**2)) * x ** (-2 / 3)
        return p

    range_start, range_end, estimated_a, estimated_m, estimated_d, sample_data = _a

    def bekkers_cdf(x, a, m, d, range_start, range_end):
        values = []
        for value in x:
            integral = integrate.quad(
                lambda k: bekkers(k, a, m, d), range_start, value
            )[0]
            normalized = (
                integral
                / integrate.quad(lambda k: bekkers(k, a, m, d), range_start, range_end)[
                    0
                ]
            )
            values.append(normalized)
        return np.array(values)

    s, p_value = stats.kstest(
        sample_data,
        lambda x: bekkers_cdf(
            x, estimated_a, estimated_m, estimated_d, range_start, range_end
        ),
    )

    if p_value >= 0.05:
        result = False
    else:
        result = True
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
