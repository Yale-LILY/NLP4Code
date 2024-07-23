import tensorflow as tf


def g(x, y, z):
    return tf.gather_nd(x, [y, z])


def define_test_input(args):
    if args.test_case == 1:
        x = [[1, 2, 3], [4, 5, 6]]
        y = [0, 1]
        z = [1, 2]
        x = tf.constant(x)
        y = tf.constant(y)
        z = tf.constant(z)
    if args.test_case == 2:
        x = [[1, 2, 3], [4, 5, 6]]
        y = [0, 1]
        z = [1, 0]
        x = tf.constant(x)
        y = tf.constant(y)
        z = tf.constant(z)
    return x, y, z


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    x, y, z = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((x, y, z), f)

    result = g(x, y, z)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
