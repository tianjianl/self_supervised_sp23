import matplotlib.pyplot as plt
import numpy as np

# load from importances.txt where each line is a printed list of importances
importances = [[] for _ in range(24)]

f = open('importances.txt', 'r')
cmap = plt.cm.get_cmap('viridis')
plt.figure(figsize=(10, 10))
for index, line in enumerate(f):
    line = line[1:-2].split(', ')
    line = [float(x) for x in line]
    # if index is large, use deeper color to show the trend
    color = cmap(index / 220) 
    plt.plot(line, color=color)

plt.xlabel('Layer shallow -> deep')
plt.ylabel('Importance')
plt.savefig('importances.pdf')
    