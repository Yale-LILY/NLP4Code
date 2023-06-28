import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    {
        "celltype": ["foo", "bar", "qux", "woz"],
        "s1": [5, 9, 1, 7],
        "s2": [12, 90, 13, 87],
    }
)

# For data in df, make a bar plot of s1 and s1 and use celltype as the xlabel
# Make the x-axis tick labels horizontal
# SOLUTION START
df = df[["celltype", "s1", "s2"]]
df.set_index(["celltype"], inplace=True)
df.plot(kind="bar", alpha=0.75, rot=0)
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
