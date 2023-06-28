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

data = {
    "reports": [4, 24, 31, 2, 3],
    "coverage": [35050800, 54899767, 57890789, 62890798, 70897871],
}
df = pd.DataFrame(data)
sns.factorplot(y="coverage", x="reports", kind="bar", data=df, label="Total")

# do not use scientific notation in the y axis ticks labels
# SOLUTION START
plt.ticklabel_format(style="plain", axis="y")
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
