import tensorflow as tf


def g(input):
    tf.compat.v1.disable_eager_execution()
    ds = tf.data.Dataset.from_tensor_slices(input)
    ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices([x, x + 1, x + 2]))
    element = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    result = []
    with tf.compat.v1.Session() as sess:
        for _ in range(9):
            result.append(sess.run(element))
    return result


def define_test_input(args):
    if args.test_case == 1:
        tf.compat.v1.disable_eager_execution()
        input = [10, 20, 30]
    elif args.test_case == 2:
        tf.compat.v1.disable_eager_execution()
        input = [20, 40, 60]
    return input


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    input = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(input, f)

    result = g(input)
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("ans"):
        os.mkdir("ans")
    if not os.path.exists("result"):
        os.mkdir("result")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
