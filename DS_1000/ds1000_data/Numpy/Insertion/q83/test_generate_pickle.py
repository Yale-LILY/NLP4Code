import numpy as np
import pandas as pd
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        grades = np.array(
            (
                93.5,
                93,
                60.8,
                94.5,
                82,
                87.5,
                91.5,
                99.5,
                86,
                93.5,
                92.5,
                78,
                76,
                69,
                94.5,
                89.5,
                92.8,
                78,
                65.5,
                98,
                98.5,
                92.3,
                95.5,
                76,
                91,
                95,
                61,
            )
        )
        eval = np.array([88, 87, 62])

    elif args.test_case == 2:
        np.random.seed(42)
        grades = (np.random.rand(50) - 0.5) * 100
        eval = np.random.randint(10, 90, (5,))

    return grades, eval


def ans1(_a):
    grades, eval = _a

    def ecdf_result(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / float(len(xs))
        return xs, ys

    resultx, resulty = ecdf_result(grades)
    result = np.zeros_like(eval, dtype=float)
    for i, element in enumerate(eval):
        if element < resultx[0]:
            result[i] = 0
        elif element >= resultx[-1]:
            result[i] = 1
        else:
            result[i] = resulty[(resultx > element).argmax() - 1]
    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(copy.deepcopy(a))

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
