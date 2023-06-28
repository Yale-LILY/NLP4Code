import tensorflow as tf
import numpy as np


def g(A):
    return tf.math.reciprocal(A)


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(10)
        A = tf.constant([-0.5, -0.1, 0, 0.1, 0.5, 2], dtype=tf.float32)
    return A


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    A = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(A, f)

    result = g(A)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
