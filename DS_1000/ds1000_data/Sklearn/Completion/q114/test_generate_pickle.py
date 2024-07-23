import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "items": ["1fgg", "2hhj", "3jkl"],
                "description": ["abcd ty", "abc r", "r df"],
            }
        )
    elif args.test_case == 2:
        df = pd.DataFrame(
            {
                "items": ["1fgg", "2hhj", "3jkl", "4dsd"],
                "description": [
                    "Chinese Beijing Chinese",
                    "Chinese Chinese Shanghai",
                    "Chinese Macao",
                    "Tokyo Japan Chinese",
                ],
            }
        )
    return df


def ans1(df):
    tfidf = TfidfVectorizer()
    response = tfidf.fit_transform(df["description"]).toarray()
    tf_idf = response
    cosine_similarity_matrix = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(len(df)):
            cosine_similarity_matrix[i, j] = cosine_similarity(
                [tf_idf[i, :]], [tf_idf[j, :]]
            )
    return cosine_similarity_matrix


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    ans = ans1(df)
    # print(ans)
    # np.testing.assert_equal(ans, ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
