import pandas as pd
import numpy as np

import numpy as np


def g(df):
    time = df.time.tolist()
    car = df.car.tolist()
    farmost_neighbour = []
    euclidean_distance = []
    for i in range(len(df)):
        n = 0
        d = 0
        for j in range(len(df)):
            if (
                df.loc[i, "time"] == df.loc[j, "time"]
                and df.loc[i, "car"] != df.loc[j, "car"]
            ):
                t = np.sqrt(
                    ((df.loc[i, "x"] - df.loc[j, "x"]) ** 2)
                    + ((df.loc[i, "y"] - df.loc[j, "y"]) ** 2)
                )
                if t >= d:
                    d = t
                    n = df.loc[j, "car"]
        farmost_neighbour.append(n)
        euclidean_distance.append(d)
    return pd.DataFrame(
        {
            "time": time,
            "car": car,
            "farmost_neighbour": farmost_neighbour,
            "euclidean_distance": euclidean_distance,
        }
    )


def define_test_input(args):
    if args.test_case == 1:
        time = [0, 0, 0, 1, 1, 2, 2]
        x = [216, 218, 217, 280, 290, 130, 132]
        y = [13, 12, 12, 110, 109, 3, 56]
        car = [1, 2, 3, 1, 3, 4, 5]
        df = pd.DataFrame({"time": time, "x": x, "y": y, "car": car})
    if args.test_case == 2:
        time = [0, 0, 0, 1, 1, 2, 2]
        x = [219, 219, 216, 280, 290, 130, 132]
        y = [15, 11, 14, 110, 109, 3, 56]
        car = [1, 2, 3, 1, 3, 4, 5]
        df = pd.DataFrame({"time": time, "x": x, "y": y, "car": car})
    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    result = g(df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
