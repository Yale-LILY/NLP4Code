import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.DataFrame(
    {
        "id": ["1", "2", "1", "2", "2"],
        "x": [123, 22, 356, 412, 54],
        "y": [120, 12, 35, 41, 45],
    }
)

# Use seaborn to make a pairplot of data in `df` using `x` for x_vars, `y` for y_vars, and `id` for hue
# Hide the legend in the output figure
# SOLUTION START
g = sns.pairplot(df, x_vars=["x"], y_vars=["y"], hue="id")
g._legend.remove()
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
