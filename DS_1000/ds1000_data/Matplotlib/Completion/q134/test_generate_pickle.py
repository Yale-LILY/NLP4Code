import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt
import numpy as np

xvec = np.linspace(-5.0, 5.0, 100)
x, y = np.meshgrid(xvec, xvec)
z = -np.hypot(x, y)
plt.contourf(x, y, z)

# draw x=0 and y=0 axis in my contour plot with white color
# SOLUTION START
plt.axhline(0, color="white")
plt.axvline(0, color="white")
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
