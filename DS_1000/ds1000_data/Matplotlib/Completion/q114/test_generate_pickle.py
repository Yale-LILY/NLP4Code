import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt
import numpy as np

# Specify the values of blue bars (height)
blue_bar = (23, 25, 17)
# Specify the values of orange bars (height)
orange_bar = (19, 18, 14)

# Plot the blue bar and the orange bar side-by-side in the same bar plot.
# Make  sure the bars don't overlap with each other.
# SOLUTION START
# Position of bars on x-axis
ind = np.arange(len(blue_bar))

# Figure size
plt.figure(figsize=(10, 5))

# Width of a bar
width = 0.3
plt.bar(ind, blue_bar, width, label="Blue bar label")
plt.bar(ind + width, orange_bar, width, label="Orange bar label")
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
