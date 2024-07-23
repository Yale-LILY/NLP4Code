import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def define_test_input(args):
    if args.test_case == 1:
        words = "Hello @friend, this is a good day. #good."
    elif args.test_case == 2:
        words = (
            "ha @ji me te no ru bu ru wa, @na n te ko to wa na ka tsu ta wa. wa ta shi da ke no mo na ri za, "
            "mo u to kku ni #de a t te ta ka ra"
        )
    return words


def ans1(words):
    count = CountVectorizer(
        lowercase=False, token_pattern="[a-zA-Z0-9$&+:;=@#|<>^*()%-]+"
    )
    vocabulary = count.fit_transform([words])
    feature_names = count.get_feature_names_out()
    return feature_names


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    words = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(words, f)

    ans = ans1(words)
    # print(ans)
    # np.testing.assert_equal(ans, ans)
    # print(ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
