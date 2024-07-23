import tensorflow as tf
import numpy as np


def g(a):
    return tf.squeeze(a)


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(10)
        a = tf.constant(np.random.rand(5, 10, 1, 51))
    if args.test_case == 2:
        np.random.seed(10)
        a = tf.constant(np.random.rand(5, 2, 1, 5))
    return a


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    result = g(a)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
