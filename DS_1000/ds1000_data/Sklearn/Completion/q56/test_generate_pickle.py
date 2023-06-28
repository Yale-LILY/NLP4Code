import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def define_test_input(args):
    if args.test_case == 1:
        document1 = [
            "Education is the process of learning and acquiring knowledge at an educational institution. It can be divided into three levels: primary, secondary, and tertiary."
            "Primary education generally refers to the first stage of schooling, which is typically compulsory and free in most countries. It usually lasts for six or seven years, from ages five or six to eleven or twelve."
            "Secondary education usually lasts for three or four years, from ages eleven or twelve to fifteen or sixteen. In some countries, it is compulsory, while in others it is not."
            "Tertiary education, also known as post-secondary education, is the stage of education that comes after secondary school. It can be divided into two types: university education and vocational education."
            "University education typically lasts for four years, from ages eighteen to twenty-two. It is usually divided into two parts: undergraduate and graduate."
            "Vocational education is training for a specific trade or profession. It can be either formal or informal. Formal vocational education is typically provided by vocational schools, while informal vocational education is provided by apprenticeships and on-the-job training."
        ]
        document2 = [
            "The purpose of education is to prepare individuals for successful futures. Whether that means getting a good job, being a responsible citizen, or becoming a lifelong learner, education is the key to a bright future."
            "There are many different types of educational institutions, each with its own unique purpose. Primary and secondary schools are the most common, but there are also institutions for special needs education, higher education, and adult education."
            "All educational institutions share the common goal of providing quality education to their students. However, the methods and curriculum used to achieve this goal can vary greatly."
            "Some educational institutions focus on academic knowledge, while others place more emphasis on practical skills. Some use traditional teaching methods, while others use more innovative approaches."
            "The type of education an individual receives should be based on their needs and goals. There is no one-size-fits-all approach to education, and what works for one person may not work for another."
        ]
        document3 = [
            "Education is a fundamental human right. It is essential for the development of individuals and societies. "
            "Access to education should be universal and inclusive. "
            "All individuals, regardless of race, gender, or social status, should have the opportunity to receive an education. Education is a tool that can be used to empower individuals and improve the quality of their lives. "
            "It can help people to develop new skills and knowledge, and to achieve their full potential. Education is also a key factor in the development of societies. "
            "It can help to create more stable and prosperous societies."
        ]
        document4 = [
            "The quality of education is a key factor in the development of individuals and societies."
            "A high-quality education can help individuals to develop their skills and knowledge, and to achieve their full potential. It can also help to create more stable and prosperous societies."
            "There are many factors that contribute to the quality of education. These include the qualifications of teachers, the resources available, and the curriculum being taught."
            "Improving the quality of education is an important goal for all educational institutions. By providing quality education, we can empower individuals and help to create a better world for everyone."
        ]
        # document5 = [
        #     "Education is an important investment in the future.\n\n"
        #     "By providing quality education, we can empower individuals and help to create a better world for everyone.\n\nA high-quality education can help individuals to develop their skills and knowledge, and to achieve their full potential. It can also help to create more stable and prosperous societies.\n\n"
        #     "There are many factors that contribute to the quality of education. These include the qualifications of teachers, the resources available, and the curriculum being taught.\n\n"
        #     "Improving the quality of education is an important goal for all educational institutions. By investing in education, we can make a difference in the world."
        # ]
        documents = []
        documents.extend(document1)
        documents.extend(document2)
        documents.extend(document3)
        documents.extend(document4)
        # documents.extend(document5)
        query1 = ["What are the benefits of education?"]
        query2 = ["What are the different branches of the military?"]
        query3 = ["What is the definition of science?"]
        query4 = ["What are the different types of education?"]
        queries = []
        queries.extend(query1)
        queries.extend(query2)
        queries.extend(query3)
        queries.extend(query4)
    return queries, documents


def ans1(queries, documents):
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(documents)

    cosine_similarities_of_queries = []
    for query in queries:
        query_tfidf = tfidf.transform([query])
        cosine_similarities_of_queries.append(
            cosine_similarity(query_tfidf, tfidf.transform(documents)).flatten()
        )
    return cosine_similarities_of_queries


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    queries, documents = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump((queries, documents), f)

    ans = ans1(queries, documents)
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
