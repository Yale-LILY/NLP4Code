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
z = np.arange(10)
a = np.arange(10)

# plot y over x and z over a in two different subplots
# Set "Y and Z" as a main title above the two subplots
# SOLUTION START
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(x, y)
axes[1].plot(a, z)
plt.suptitle("Y and Z")
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
