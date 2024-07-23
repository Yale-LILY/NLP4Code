import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import random

from scipy import stats
import random
import numpy as np


def define_test_input(args):
    def poisson_simul(rate, T):
        time = random.expovariate(rate)
        times = [0]
        while times[-1] < T:
            times.append(time + times[-1])
            time = random.expovariate(rate)
        return times[1:]

    if args.test_case == 1:
        random.seed(42)
        rate = 1
        T = 100
        times = poisson_simul(rate, T)
    elif args.test_case == 2:
        random.seed(45)
        rate = 1
        T = 500
        times = poisson_simul(rate, T)
    return rate, T, times


def ans1(_a):
    rate, T, times = _a
    res = stats.kstest(times, stats.uniform(loc=0, scale=T).cdf)

    if res[1] < 0.05:
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
