import pickle
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--test_case", type=int, default=1)
args = parser.parse_args()


import matplotlib.pyplot as plt
import numpy

xlabels = list("ABCD")
ylabels = list("CDEF")
rand_mat = numpy.random.rand(4, 4)

# Plot of heatmap with data in rand_mat and use xlabels for x-axis labels and ylabels as the y-axis labels
# Make the x-axis tick labels appear on top of the heatmap and invert the order or the y-axis labels (C to F from top to bottom)
# SOLUTION START
plt.pcolor(rand_mat)
plt.xticks(numpy.arange(0.5, len(xlabels)), xlabels)
plt.yticks(numpy.arange(0.5, len(ylabels)), ylabels)
ax = plt.gca()
ax.invert_yaxis()
ax.xaxis.tick_top()
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
