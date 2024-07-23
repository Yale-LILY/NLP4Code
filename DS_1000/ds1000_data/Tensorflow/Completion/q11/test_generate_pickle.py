import tensorflow as tf


def g(lengths):
    lengths = [8 - x for x in lengths]
    lengths_transposed = tf.expand_dims(lengths, 1)
    range = tf.range(0, 8, 1)
    range_row = tf.expand_dims(range, 0)
    mask = tf.less(range_row, lengths_transposed)
    result = tf.where(~mask, tf.ones([4, 8]), tf.zeros([4, 8]))
    return result


def define_test_input(args):
    if args.test_case == 1:
        lengths = [4, 3, 5, 2]
    if args.test_case == 2:
        lengths = [2, 3, 4, 5]
    return lengths


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    lengths = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(lengths, f)

    result = g(lengths)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
