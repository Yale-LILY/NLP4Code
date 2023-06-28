import tensorflow as tf


def g(x):
    non_zero = tf.cast(x != 0, tf.float32)
    y = tf.reduce_sum(x, axis=-2) / tf.reduce_sum(non_zero, axis=-2)
    return y


def define_test_input(args):
    if args.test_case == 1:
        x = [
            [
                [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
                [[1, 2, 3], [2, 0, 4], [3, 4, 5]],
                [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 3], [1, 2, 3], [0, 0, 0]],
            ],
            [
                [[1, 2, 3], [0, 1, 0], [0, 0, 0]],
                [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
                [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            ],
        ]
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if args.test_case == 2:
        x = [
            [
                [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
                [[1, 2, 3], [2, 0, 4], [3, 4, 5]],
                [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 3], [1, 2, 3], [0, 0, 0]],
            ],
            [
                [[1, 2, 3], [0, 1, 0], [0, 0, 0]],
                [[1, 2, 3], [2, 3, 4], [0, 0, 0]],
                [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 2, 3], [1, 2, 3]],
            ],
        ]
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    return x


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    x = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(x, f)

    result = g(x)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
