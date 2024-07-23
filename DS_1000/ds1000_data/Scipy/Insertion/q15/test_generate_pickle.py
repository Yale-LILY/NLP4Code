import numpy as np
import pandas as pd
import scipy
from scipy import sparse, misc


from scipy import misc
from scipy.ndimage import rotate
import numpy as np


def define_test_input(args):
    if args.test_case == 1:
        data_orig = misc.face()
        x0, y0 = 580, 300  # left eye; (xrot,yrot) should point there
        np.random.seed(42)
        angle = np.random.randint(1, 360)
    return data_orig, x0, y0, angle


def ans1(_a):
    data_orig, x0, y0, angle = _a

    def rot_ans(image, xy, angle):
        im_rot = rotate(image, angle)
        org_center = (np.array(image.shape[:2][::-1]) - 1) / 2.0
        rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) / 2.0
        org = xy - org_center
        a = np.deg2rad(angle)
        new = np.array(
            [
                org[0] * np.cos(a) + org[1] * np.sin(a),
                -org[0] * np.sin(a) + org[1] * np.cos(a),
            ]
        )
        return im_rot, new + rot_center

    data_rot, (xrot, yrot) = rot_ans(data_orig, np.array([x0, y0]), angle)

    return [data_rot, (xrot, yrot)]


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
