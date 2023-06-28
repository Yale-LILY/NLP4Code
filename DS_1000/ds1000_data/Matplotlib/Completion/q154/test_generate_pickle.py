import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.random((10, 10))
from matplotlib import gridspec

nrow = 2
ncol = 2

fig = plt.figure(figsize=(ncol + 1, nrow + 1))

# Make a 2x2 subplots with fig and plot x in each subplot as an image
# Remove the space between each subplot and make the subplot adjacent to each other
# Remove the axis ticks from each subplot
# SOLUTION START
gs = gridspec.GridSpec(
    nrow,
    ncol,
    wspace=0.0,
    hspace=0.0,
    top=1.0 - 0.5 / (nrow + 1),
    bottom=0.5 / (nrow + 1),
    left=0.5 / (ncol + 1),
    right=1 - 0.5 / (ncol + 1),
)

for i in range(nrow):
    for j in range(ncol):
        ax = plt.subplot(gs[i, j])
        ax.imshow(x)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
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
