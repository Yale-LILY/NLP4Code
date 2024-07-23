import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
axes = axes.flatten()

for ax in axes:
    ax.set_ylabel(r"$\ln\left(\frac{x_a-x_b}{x_a-x_c}\right)$")
    ax.set_xlabel(r"$\ln\left(\frac{x_a-x_d}{x_a-x_e}\right)$")

plt.show()
plt.clf()

# Copy the previous plot but adjust the subplot padding to have enough space to display axis labels
# SOLUTION START
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
axes = axes.flatten()

for ax in axes:
    ax.set_ylabel(r"$\ln\left(\frac{x_a-x_b}{x_a-x_c}\right)$")
    ax.set_xlabel(r"$\ln\left(\frac{x_a-x_d}{x_a-x_e}\right)$")

plt.tight_layout()
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
