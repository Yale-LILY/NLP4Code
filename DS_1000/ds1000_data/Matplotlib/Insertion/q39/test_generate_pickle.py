import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x)
y2 = np.cos(x)

# plot x vs y1 and x vs y2 in two subplots
# remove the frames from the subplots
# SOLUTION START
fig, (ax1, ax2) = plt.subplots(nrows=2, subplot_kw=dict(frameon=False))

plt.subplots_adjust(hspace=0.0)
ax1.grid()
ax2.grid()

ax1.plot(x, y1, color="r")
ax2.plot(x, y2, color="b", linestyle="--")
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
