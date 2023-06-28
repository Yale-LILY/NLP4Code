import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


from numpy import *
import math
import matplotlib
import matplotlib.pyplot as plt

t = linspace(0, 2 * math.pi, 400)
a = sin(t)
b = cos(t)
c = a + b

# Plot a, b, c in the same figure
# SOLUTION START
plt.plot(t, a, t, b, t, c)
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
