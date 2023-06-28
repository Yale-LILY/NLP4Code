import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# def define_test_input(args):
#     if args.test_case == 1:
#         X = np.array([[-1, 2], [-0.5, 6]])
#     return X


def create_data():
    df = pd.DataFrame(
        {
            "Name": [
                "T-Rex",
                "Crocodile",
                "Lion",
                "Bear",
                "Tiger",
                "Hyena",
                "Jaguar",
                "Cheetah",
                "KomodoDragon",
            ],
            "teethLength": [12, 4, 2.7, 3.6, 3, 0.27, 2, 1.5, 0.4],
            "weight": [15432, 2400, 416, 600, 260, 160, 220, 154, 150],
            "length": [40, 23, 9.8, 7, 12, 5, 5.5, 4.9, 8.5],
            "hieght": [20, 1.6, 3.9, 3.35, 3, 2, 2.5, 2.9, 1],
            "speed": [33, 8, 50, 40, 40, 37, 40, 70, 13],
            "Calorie Intake": [40000, 2500, 7236, 20000, 7236, 5000, 5000, 2200, 1994],
            "Bite Force": [12800, 3700, 650, 975, 1050, 1100, 1350, 475, 240],
            "Prey Speed": [20, 30, 35, 0, 37, 20, 15, 56, 24],
            "PreySize": [19841, 881, 1300, 0, 160, 40, 300, 185, 110],
            "EyeSight": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Smell": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Class": [
                "Primary Hunter",
                "Primary Hunter",
                "Primary Hunter",
                "Primary Scavenger",
                "Primary Hunter",
                "Primary Scavenger",
                "Primary Hunter",
                "Primary Hunter",
                "Primary Scavenger",
            ],
        }
    )

    filename = "animalData.csv"
    df.to_csv(filename, index=False, sep=",")


def ans1():
    filename = "animalData.csv"
    dataframe = pd.read_csv(filename, dtype="category")
    # dataframe = df
    # Git rid of the name of the animal
    # And change the hunter/scavenger to 0/1
    dataframe = dataframe.drop(["Name"], axis=1)
    cleanup = {"Class": {"Primary Hunter": 0, "Primary Scavenger": 1}}
    dataframe.replace(cleanup, inplace=True)

    # Seperating the data into dependent and independent variables
    X = dataframe.iloc[:, 0:-1].astype(float)
    y = dataframe.iloc[:, -1]

    logReg = LogisticRegression()
    ### END SOLUTION
    logReg.fit(X[:None], y)
    predict = logReg.predict(X)
    return predict


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    # X = define_test_input(args)
    #
    # if not os.path.exists("input"):
    #     os.mkdir("input")
    # with open('input/input{}.pkl'.format(args.test_case), 'wb') as f:
    #     pickle.dump(X, f)
    create_data()

    ans = ans1()
    # print(ans)
    # np.testing.assert_equal(ans, ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
