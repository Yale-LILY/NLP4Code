import tensorflow as tf
import numpy as np


def g(A, B):
    return tf.constant(np.einsum("ikm, jkm-> ijk", A, B))


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(10)
        A = tf.constant(np.random.randint(low=0, high=5, size=(10, 20, 30)))
        B = tf.constant(np.random.randint(low=0, high=5, size=(10, 20, 30)))

    return A, B


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    A, B = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((A, B), f)

    result = g(A, B)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
