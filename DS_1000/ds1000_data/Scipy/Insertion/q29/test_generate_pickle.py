import numpy as np
import pandas as pd
import scipy
from scipy import sparse, ndimage


import numpy as np
from scipy import ndimage


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(10)
        gen = np.random.RandomState(0)
        img = gen.poisson(2, size=(512, 512))
        img = ndimage.gaussian_filter(img.astype(np.double), (30, 30))
        img -= img.min()
        img /= img.max()
    return img


def ans1(_a):
    img = _a
    threshold = 0.75
    blobs = img > threshold
    labels, nlabels = ndimage.label(blobs)
    r, c = np.vstack(ndimage.center_of_mass(img, labels, np.arange(nlabels) + 1)).T
    # find their distances from the top-left corner
    d = np.sqrt(r * r + c * c)
    result = sorted(d)

    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(a)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
