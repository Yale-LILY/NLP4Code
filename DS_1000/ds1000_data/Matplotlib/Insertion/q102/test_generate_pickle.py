import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt
import numpy as np

data = np.random.random((10, 10))

# Set xlim and ylim to be between 0 and 10
# Plot a heatmap of data in the rectangle where right is 5, left is 1, bottom is 1, and top is 4.
# SOLUTION START
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.imshow(data, extent=[1, 5, 1, 4])
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
