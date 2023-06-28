import numpy as np
import pandas as pd
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        a = np.ones((3, 3))
        low_index = -1
        high_index = 2
    elif args.test_case == 2:
        a = np.ones((5, 5)) * 2
        low_index = 1
        high_index = 6
    elif args.test_case == 3:
        a = np.ones((5, 5))
        low_index = 2
        high_index = 7
    return a, low_index, high_index


def ans1(_a):
    a, low_index, high_index = _a

    def fill_crop(img, pos, crop):
        img_shape, pos, crop_shape = (
            np.array(img.shape),
            np.array(pos),
            np.array(crop.shape),
        )
        end = pos + crop_shape
        # Calculate crop slice positions
        crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
        crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
        crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
        # Calculate img slice positions
        pos = np.clip(pos, a_min=0, a_max=img_shape)
        end = np.clip(end, a_min=0, a_max=img_shape)
        img_slices = (slice(low, high) for low, high in zip(pos, end))
        crop[tuple(crop_slices)] = img[tuple(img_slices)]
        return crop

    result = fill_crop(
        a,
        [low_index, low_index],
        np.zeros((high_index - low_index, high_index - low_index)),
    )

    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(copy.deepcopy(a))

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
