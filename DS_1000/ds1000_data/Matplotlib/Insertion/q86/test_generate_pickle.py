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
a = np.arange(10)
z = np.arange(10)

# Plot y over x and a over z in two side-by-side subplots.
# Label them "y" and "a" and make a single figure-level legend using the figlegend function
# SOLUTION START
fig, axs = plt.subplots(1, 2)
axs[0].plot(x, y, label="y")
axs[1].plot(z, a, label="a")
plt.figlegend(["y", "a"])
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
