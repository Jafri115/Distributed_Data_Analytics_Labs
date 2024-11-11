import numpy as np
from matplotlib import pyplot as plt


speedupList = []

with open('TimeforTraining.txt', 'r') as f:
    Lines = f.readlines()
    timeList = []
    print(Lines)
    for line in Lines:
        timeList.append([float(x) for x in line.strip().split(',')])

print(np.array( timeList))
workers = np.array(timeList)[:,0]
executionTime = np.array(timeList)[:,1]
accuracy = np.array(timeList)[:,2]

ts = executionTime[0]
sp = []

for i in range(len(workers)):
    sp.append(ts/executionTime[i])

    
plt.plot(workers, sp,marker='o')
plt.title('Speedup Graph')
plt.xlabel('Processes')
plt.ylabel('S P')
plt.grid()
plt.show()

plt.plot(workers, accuracy,marker='o')
plt.title('Accuracy Graph')
plt.xlabel('Processes')
plt.ylabel('Accuracy')
plt.grid()
plt.show()