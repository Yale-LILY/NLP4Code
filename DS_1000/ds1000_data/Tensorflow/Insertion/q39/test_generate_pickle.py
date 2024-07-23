import tensorflow as tf


def g(a):
    return tf.argmin(a, axis=0)


def define_test_input(args):
    if args.test_case == 1:
        a = tf.constant(
            [
                [0.3232, -0.2321, 0.2332, -0.1231, 0.2435, 0.6728],
                [0.2323, -0.1231, -0.5321, -0.1452, 0.5435, 0.1722],
                [0.9823, -0.1321, -0.6433, 0.1231, 0.023, 0.0711],
            ]
        )
    if args.test_case == 2:
        a = tf.constant(
            [
                [0.3232, -0.2321, 0.2332, -0.1231, 0.2435],
                [0.2323, -0.1231, -0.5321, -0.1452, 0.5435],
                [0.9823, -0.1321, -0.6433, 0.1231, 0.023],
            ]
        )
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
