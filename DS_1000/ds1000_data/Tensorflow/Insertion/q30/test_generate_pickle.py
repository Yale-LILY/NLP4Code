import tensorflow as tf


def g(x):
    return [tf.compat.as_str_any(a) for a in x]


def define_test_input(args):
    if args.test_case == 1:
        x = [
            b"\xd8\xa8\xd9\x85\xd8\xb3\xd8\xa3\xd9\x84\xd8\xa9",
            b"\xd8\xa5\xd9\x86\xd8\xb4\xd8\xa7\xd8\xa1",
            b"\xd9\x82\xd8\xb6\xd8\xa7\xd8\xa1",
            b"\xd8\xac\xd9\x86\xd8\xa7\xd8\xa6\xd9\x8a",
            b"\xd8\xaf\xd9\x88\xd9\x84\xd9\x8a",
        ]

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
