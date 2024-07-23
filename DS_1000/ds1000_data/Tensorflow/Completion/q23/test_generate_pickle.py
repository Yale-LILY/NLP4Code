import tensorflow as tf


def g(a, b):
    return tf.reduce_sum(tf.square(tf.subtract(a, b)), 0)


def define_test_input(args):
    if args.test_case == 1:
        a = tf.constant([[1, 1, 1], [1, 1, 1]])
        b = tf.constant([[0, 0, 0], [1, 1, 1]])
    if args.test_case == 2:
        a = tf.constant([[0, 1, 1], [1, 0, 1]])
        b = tf.constant([[0, 0, 0], [1, 1, 1]])
    return a, b


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a, b = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((a, b), f)

    result = g(a, b)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
