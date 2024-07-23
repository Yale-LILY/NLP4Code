import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def define_test_input(args):
    if args.test_case == 1:
        text = '''
        "I'm so tired of this," she said, "I can't take it anymore!"
        "I know how you feel," he replied, "but you have to stay strong."
        "I don't know if I can," she said, her voice trembling.
        "You can do it," he said, "I know you can."
        "But what if I can't?" she said, her eyes filling with tears.
        "You have to try," he said, "You can't give up."
        "I don't know if I can," she said, her voice shaking.
        "Yes, you can," he said, "I believe in you."'''
    elif args.test_case == 2:
        text = """
        A: So how was your day today?
        B: It was okay, I guess. I woke up late and had to rush to get ready for work.
        A: That sounds like a pain. I hate waking up late.
        B: Yeah, it's not my favorite thing either. But at least I had a good breakfast to start the day.
        A: That's true. Breakfast is the most important meal of the day.
        B: Absolutely. I always make sure to eat a healthy breakfast before starting my day.
        A: That's a good idea. I should start doing that too.
        B: Yeah, you should definitely try it. I think you'll find that you have more energy throughout the day.
        A: I'll definitely give it a try. Thanks for the suggestion.
        B: No problem. I'm always happy to help out where I can.
        A: So what did you do after work today?
        B: I went to the gym and then came home and made dinner.
        A: That sounds like a good day. I wish I had time to go to the gym more often.
        B: Yeah, it's definitely important to make time for exercise. I try to go at least three times a week.
        A: That's a good goal. I'm going to try to make it to the gym at least twice a week from now on.
        B: That's a great idea. I'm sure you'll see a difference in your energy levels.
        A: I hope so. I'm getting kind of tired of being tired all the time.
        B: I know how you feel. But I think making some changes in your lifestyle, like going to the gym more often, will definitely help.
        A: I hope you're right. I'm getting kind of sick of my current routine.
        B: I know how you feel. Sometimes it's just good to mix things up a bit.
        A: I think you're right. I'm going to try to make some changes starting next week.
        B: That's a great idea. I'm sure you'll see a difference in no time.
        """
    return text


def ans1(text):
    vent = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'")
    transformed_text = vent.fit_transform([text])
    return transformed_text


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    text = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(text, f)

    ans = ans1(text)
    # print(ans)
    # print(ans.toarray())
    # np.testing.assert_equal(ans.toarray(), ans.toarray())
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
