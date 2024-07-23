import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn

seaborn.set(style="ticks")

numpy.random.seed(0)
N = 37
_genders = ["Female", "Male", "Non-binary", "No Response"]
df = pandas.DataFrame(
    {
        "Height (cm)": numpy.random.uniform(low=130, high=200, size=N),
        "Weight (kg)": numpy.random.uniform(low=30, high=100, size=N),
        "Gender": numpy.random.choice(_genders, size=N),
    }
)

# make seaborn relation plot and color by the gender field of the dataframe df
# SOLUTION START
seaborn.relplot(
    data=df, x="Weight (kg)", y="Height (cm)", hue="Gender", hue_order=_genders
)
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
