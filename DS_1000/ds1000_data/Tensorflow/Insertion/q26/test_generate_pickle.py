import tensorflow as tf


def g(x, row, col):
    index = [[row[i], col[i]] for i in range(len(col))]
    return tf.gather_nd(x, index)


def define_test_input(args):
    if args.test_case == 1:
        x = [[1, 2, 3], [4, 5, 6]]
        row = [0, 0]
        col = [1, 2]
        x = tf.constant(x)
        row = tf.constant(row)
        col = tf.constant(col)
    if args.test_case == 2:
        x = [[1, 2, 3], [4, 5, 6]]
        row = [1, 0]
        col = [1, 2]
        x = tf.constant(x)
        row = tf.constant(row)
        col = tf.constant(col)
    return x, row, col


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    x, row, col = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((x, row, col), f)

    result = g(x, row, col)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
