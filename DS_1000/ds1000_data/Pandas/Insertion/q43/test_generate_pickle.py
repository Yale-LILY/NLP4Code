import pandas as pd
import numpy as np


def g(df):
    df.columns = np.concatenate([df.columns[0:1], df.iloc[0, 1:2], df.columns[2:]])
    df = df.iloc[1:].reset_index(drop=True)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "Nanonose": ["Sample type", "Water", "Water", "Water", "Water"],
                "Unnamed: 1": ["Concentration", 9200, 9200, 9200, 4600],
                "A": [
                    np.nan,
                    95.5,
                    94.5,
                    92.0,
                    53.0,
                ],
                "B": [np.nan, 21.0, 17.0, 16.0, 7.5],
                "C": [np.nan, 6.0, 5.0, 3.0, 2.5],
                "D": [np.nan, 11.942308, 5.484615, 11.057692, 3.538462],
                "E": [np.nan, 64.134615, 63.205769, 62.586538, 35.163462],
                "F": [np.nan, 21.498560, 19.658560, 19.813120, 6.876207],
                "G": [np.nan, 5.567840, 4.968000, 5.192480, 1.641724],
                "H": [np.nan, 1.174135, 1.883444, 0.564835, 0.144654],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "Nanonose": ["type of Sample", "Water", "Water", "Water", "Water"],
                "Unnamed: 1": ["concentration", 9200, 9200, 9200, 4600],
                "A": [
                    np.nan,
                    95.5,
                    94.5,
                    92.0,
                    53.0,
                ],
                "B": [np.nan, 21.0, 17.0, 16.0, 7.5],
                "C": [np.nan, 6.0, 5.0, 3.0, 2.5],
                "D": [np.nan, 11.942308, 5.484615, 11.057692, 3.538462],
                "E": [np.nan, 64.134615, 63.205769, 62.586538, 35.163462],
                "F": [np.nan, 21.498560, 19.658560, 19.813120, 6.876207],
                "G": [np.nan, 5.567840, 4.968000, 5.192480, 1.641724],
                "H": [np.nan, 1.174135, 1.883444, 0.564835, 0.144654],
            }
        )
    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=2)
    args = parser.parse_args()

    df = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    result = g(df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
