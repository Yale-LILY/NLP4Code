import tensorflow as tf


def g(labels):
    return tf.one_hot(indices=labels, depth=10, on_value=1, off_value=0, axis=-1)


def define_test_input(args):
    if args.test_case == 1:
        labels = [0, 6, 5, 4, 2]
    if args.test_case == 2:
        labels = [0, 1, 2, 3, 4]
    return labels


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    labels = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(labels, f)

    result = g(labels)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
