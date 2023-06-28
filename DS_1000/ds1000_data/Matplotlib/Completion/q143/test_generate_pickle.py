import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("exercise")

# Make catplots of scatter plots by using "time" as x, "pulse" as y, "kind" as hue, and "diet" as col
# Change the subplots titles to "Group: Fat" and "Group: No Fat"
# SOLUTION START
g = sns.catplot(x="time", y="pulse", hue="kind", col="diet", data=df)
axs = g.axes.flatten()
axs[0].set_title("Group: Fat")
axs[1].set_title("Group: No Fat")
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
