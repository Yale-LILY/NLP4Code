import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt

d = {"a": 4, "b": 5, "c": 7}
c = {"a": "red", "c": "green", "b": "blue"}

# Make a bar plot using data in `d`. Use the keys as x axis labels and the values as the bar heights.
# Color each bar in the plot by looking up the color in colors
# SOLUTION START
colors = []
for k in d:
    colors.append(c[k])
plt.bar(range(len(d)), d.values(), color=colors)
plt.xticks(range(len(d)), d.keys())
# SOLUTION END

os.makedirs("ans", exist_ok=True)
os.makedirs("input", exist_ok=True)
plt.savefig("ans/oracle_plot.png", bbox_inches="tight")
with open("input/input1.pkl", "wb") as file:
    # input is already contained in code_context.txt so we dump a dummy input object
    pickle.dump(None, file)
with open("ans/ans1.pkl", "wb") as file:
    # test does not require a reference solution object so we dump a dummpy ans object
    pickle.dump(None, file)
