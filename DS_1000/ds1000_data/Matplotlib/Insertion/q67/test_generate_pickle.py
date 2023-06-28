import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = [2.56422, 3.77284, 3.52623]
b = [0.15, 0.3, 0.45]
c = [58, 651, 393]

# make scatter plot of a over b and annotate each data point with correspond numbers in c
# SOLUTION START
fig, ax = plt.subplots()
plt.scatter(a, b)

for i, txt in enumerate(c):
    ax.annotate(txt, (a[i], b[i]))
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
