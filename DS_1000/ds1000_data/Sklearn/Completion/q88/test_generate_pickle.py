import numpy as np
import pandas as pd
import pandas.testing
from sklearn.feature_extraction.text import CountVectorizer


# def define_test_input(args):
#     if args.test_case == 1:
#         corpus = [
#             'This is the first document.',
#             'This document is the second document.',
#             'And this is the first piece of news',
#             'Is this the first document? No, it\'the fourth document',
#             "This is the second news"
#         ]
#         y = [0, 0, 1, 0, 1]
#     return corpus, y


def ans1():
    corpus = [
        "We are looking for Java developer",
        "Frontend developer with knowledge in SQL and Jscript",
        "And this is the third one.",
        "Is this the first document?",
    ]
    vectorizer = CountVectorizer(
        stop_words="english",
        binary=True,
        lowercase=False,
        vocabulary=[
            "Jscript",
            ".Net",
            "TypeScript",
            "NodeJS",
            "Angular",
            "Mongo",
            "CSS",
            "Python",
            "PHP",
            "Photoshop",
            "Oracle",
            "Linux",
            "C++",
            "Java",
            "TeamCity",
            "Frontend",
            "Backend",
            "Full stack",
            "UI Design",
            "Web",
            "Integration",
            "Database design",
            "UX",
        ],
    )

    X = vectorizer.fit_transform(corpus).toarray()
    rows, cols = X.shape
    for i in range(rows):
        for j in range(cols):
            if X[i, j] == 0:
                X[i, j] = 1
            else:
                X[i, j] = 0
    feature_names = vectorizer.get_feature_names_out()
    return feature_names, X


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    # corpus, y = define_test_input(args)
    #
    # if not os.path.exists("input"):
    #     os.mkdir("input")
    # with open('input/input{}.pkl'.format(args.test_case), 'wb') as f:
    #     pickle.dump((corpus, y), f)

    ans = ans1()
    # print(ans[1])
    # print(ans[0])
    # print(ans[1])
    # np.testing.assert_equal(ans[0], ans[0])
    # np.testing.assert_equal(ans[1], ans[1])
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
