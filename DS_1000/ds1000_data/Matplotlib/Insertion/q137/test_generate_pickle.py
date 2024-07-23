import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.arange(10)

# make 4 by 4 subplots with a figure size (5,5)
# in each subplot, plot y over x and show axis tick labels
# give enough spacing between subplots so the tick labels don't overlap
# SOLUTION START
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(5, 5))
for ax in axes.flatten():
    ax.plot(x, y)
fig.tight_layout()
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
