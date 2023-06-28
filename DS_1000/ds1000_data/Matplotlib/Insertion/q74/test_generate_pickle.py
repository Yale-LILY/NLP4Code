import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt

a, b = 1, 1
c, d = 3, 4

# draw a line that pass through (a, b) and (c, d)
# do not just draw a line segment
# set the xlim and ylim to be between 0 and 5
# SOLUTION START
plt.axline((a, b), (c, d))
plt.xlim(0, 5)
plt.ylim(0, 5)
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
