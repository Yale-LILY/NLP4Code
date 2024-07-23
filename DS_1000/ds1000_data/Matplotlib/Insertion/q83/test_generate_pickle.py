import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.random.randn(50, 4),
    index=pd.date_range("1/1/2000", periods=50),
    columns=list("ABCD"),
)
df = df.cumsum()

# make four line plots of data in the data frame
# show the data points  on the line plot
# SOLUTION START
df.plot(style=".-")
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
