import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import pandas as pd
import matplotlib.pyplot as plt

values = [[1, 2], [3, 4]]
df = pd.DataFrame(values, columns=["Type A", "Type B"], index=["Index 1", "Index 2"])

# Plot values in df with line chart
# label the x axis and y axis in this plot as "X" and "Y"
# SOLUTION START
df.plot()
plt.xlabel("X")
plt.ylabel("Y")
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
