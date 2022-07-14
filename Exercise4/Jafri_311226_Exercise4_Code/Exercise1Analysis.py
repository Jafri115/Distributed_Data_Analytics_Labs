import numpy as np
from matplotlib import pyplot as plt
epochesNumberList = [5,10,15,20,25]
timeLiist = []
for i in epochesNumberList:
    with open('timeingforDataset1\Timefor'+str(i)+'epochs.txt', 'r') as f:
        Lines = f.readlines()
        timeList = []
        for line in Lines:
            timeList.append([float(x) for x in line.strip().split(',')])
    
    workers = np.array(timeList)[:,0]
    executionTime = np.array(timeList)[:,1]

    timeLiist.append(executionTime)
    

for i in range(len(timeLiist)):
    plt.plot(workers, timeLiist[i], label='total epochs='+str(epochesNumberList[i]))
plt.title('Time Graph for different Number of Total Epochs')
plt.xlabel('Processes')
plt.ylabel('Execution Time ')
plt.legend()
plt.rcParams["figure.figsize"] = (25,10)
plt.show()