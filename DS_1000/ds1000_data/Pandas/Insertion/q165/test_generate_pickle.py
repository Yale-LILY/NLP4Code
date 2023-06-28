import pandas as pd
import numpy as np


def g(someTuple):
    return pd.DataFrame(np.column_stack(someTuple), columns=["birdType", "birdCount"])


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(123)
        birds = np.random.choice(
            ["African Swallow", "Dead Parrot", "Exploding Penguin"], size=int(5e4)
        )
        someTuple = np.unique(birds, return_counts=True)

    return someTuple


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    someTuple = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(someTuple, f)

    result = g(someTuple)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
