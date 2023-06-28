import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt

labels = ["Walking", "Talking", "Sleeping", "Working"]
sizes = [23, 45, 12, 20]
colors = ["red", "blue", "green", "yellow"]

# Make a pie chart with data in `sizes` and use `labels` as the pie labels and `colors` as the pie color.
# Bold the pie labels
# SOLUTION START
plt.pie(sizes, colors=colors, labels=labels, textprops={"weight": "bold"})
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
