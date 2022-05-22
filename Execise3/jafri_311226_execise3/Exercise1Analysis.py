import numpy as np
from matplotlib import pyplot as plt
numberOfClustersList = [1]
speedupList = []
for i in numberOfClustersList:
    with open('Timefork'+str(i)+'.txt', 'r') as f:
        Lines = f.readlines()
        timeList = []
        print(Lines)
        for line in Lines:
            timeList.append([x for x in line.strip().split(',')])
    
    print(np.array(timeList))
    workers = np.array(timeList)[:,0]
    executionTime = np.array(timeList)[:,1]

    ts = executionTime[0]
    sp = []

    for i in range(len(workers)):
        sp.append(ts/executionTime[i])
    speedupList.append(sp)
    

for i in range(len(speedupList)):
    plt.plot(workers, speedupList[i], label='Clusters='+str(numberOfClustersList[i]),marker='o')
plt.title('Speedup Graph')
plt.xlabel('Processes')
plt.ylabel('S P')
plt.legend()
plt.rcParams["figure.figsize"] = (25,10)
plt.show()