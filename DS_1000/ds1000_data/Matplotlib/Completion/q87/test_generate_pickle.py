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

df = sns.load_dataset("penguins")[
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
]

# Make 2 subplots.
# In the first subplot, plot a seaborn regression plot of "bill_depth_mm" over "bill_length_mm"
# In the second subplot, plot a seaborn regression plot of "flipper_length_mm" over "bill_length_mm"
# Do not share y axix for the subplots
# SOLUTION START
f, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.regplot(x="bill_length_mm", y="bill_depth_mm", data=df, ax=ax[0])
sns.regplot(x="bill_length_mm", y="flipper_length_mm", data=df, ax=ax[1])
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
