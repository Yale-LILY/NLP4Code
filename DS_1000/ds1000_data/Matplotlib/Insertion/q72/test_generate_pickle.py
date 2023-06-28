import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.rand(10)
y = np.random.rand(10)
bins = np.linspace(-1, 1, 100)

# Plot two histograms of x and y on a single chart with matplotlib
# Set the transparency of the histograms to be 0.5
# SOLUTION START
plt.hist(x, bins, alpha=0.5, label="x")
plt.hist(y, bins, alpha=0.5, label="y")
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
