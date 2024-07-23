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
y = np.random.rand(10)
z = np.random.rand(10)
a = np.arange(10)

# Make two subplots
# Plot y over x in the first subplot and plot z over a in the second subplot
# Label each line chart and put them into a single legend on the first subplot
# SOLUTION START
fig, ax = plt.subplots(2, 1)
(l1,) = ax[0].plot(x, y, color="red", label="y")
(l2,) = ax[1].plot(a, z, color="blue", label="z")
ax[0].legend([l1, l2], ["z", "y"])
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
