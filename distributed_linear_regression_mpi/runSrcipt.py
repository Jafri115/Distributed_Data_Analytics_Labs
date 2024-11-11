import os
workers = [1,2,4,6,7]
epochs = [20]
for e in epochs:
    for i in workers:
        os.system("mpiexec -n "+str(i)+" python .\Dataset2.py "+ str(e))

