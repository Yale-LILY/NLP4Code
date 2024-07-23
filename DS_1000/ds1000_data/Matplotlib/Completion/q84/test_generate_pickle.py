import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy as np
import matplotlib.pyplot as plt

data = [1000, 1000, 5000, 3000, 4000, 16000, 2000]

# Make a histogram of data and renormalize the data to sum up to 1
# Format the y tick labels into percentage and set y tick labels as 10%, 20%, etc.
# SOLUTION START
plt.hist(data, weights=np.ones(len(data)) / len(data))
from matplotlib.ticker import PercentFormatter

ax = plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(1))
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
