import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt
import numpy as np

box_position, box_height, box_errors = np.arange(4), np.ones(4), np.arange(1, 5)
c = ["r", "r", "b", "b"]
fig, ax = plt.subplots()
ax.bar(box_position, box_height, color="yellow")

# Plot error bars with errors specified in box_errors. Use colors in c to color the error bars
# SOLUTION START
for pos, y, err, color in zip(box_position, box_height, box_errors, c):
    ax.errorbar(pos, y, err, color=color)
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
