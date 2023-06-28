import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import seaborn as sns
import matplotlib.pylab as plt
import pandas
import numpy as np

df = pandas.DataFrame(
    {
        "a": np.arange(1, 31),
        "b": ["A",] * 10 + ["B",] * 10 + ["C",] * 10,
        "c": np.random.rand(30),
    }
)

# Use seaborn FaceGrid for rows in "b" and plot seaborn pointplots of "c" over "a"
# In each subplot, show xticks of intervals of 1 but show xtick labels with intervals of 2
# SOLUTION START
g = sns.FacetGrid(df, row="b")
g.map(sns.pointplot, "a", "c")

for ax in g.axes.flat:
    labels = ax.get_xticklabels()  # get x labels
    for i, l in enumerate(labels):
        if i % 2 == 0:
            labels[i] = ""  # skip even labels
    ax.set_xticklabels(labels)  # set new labels
# SOLUTION END

os.makedirs('ans', exist_ok=True)
os.makedirs('input', exist_ok=True)
plt.savefig('ans/oracle_plot.png', bbox_inches ='tight')
with open('input/input1.pkl', 'wb') as file:
    # input is already contained in code_context.txt so we dump a dummy input object
    pickle.dump(None, file)
with open('ans/ans1.pkl', 'wb') as file:
    # test does not require a reference solution object so we dump a dummpy ans object
    pickle.dump(None, file) 
