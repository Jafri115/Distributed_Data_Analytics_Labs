# importing libraries
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

total_worker = comm.Get_size()
worker_id = comm.Get_rank()
start_time = MPI.Wtime()  # time variable to find time at the end

n = 10**7 # size of array

root = 0  # master with id=0
small_slice = n // total_worker
if worker_id == root:  # Condition of checking master to distribute data

    v1 = np.array(np.random.randint(10, size=n))
    print("Vector 1: ", v1)
    print("Vector size : ", n)
    # slicing the vector for the number of workers

    for i in range(1, total_worker):  # for loop to send data

        #slicing vector
        start_index = (i - 1) * small_slice
        end_index = i * small_slice
        v1_slice= v1[start_index:end_index]

        # sending the chunks of array to every worker
        comm.send(v1_slice, dest=i,tag=1)


    summing = []

    #master calculation of average for last slice
    start_index = (n-small_slice)
    end_index = n
    v1_slice = v1[start_index:end_index]
    total = 0
    for i in range(small_slice):
        total += v1_slice[i] # appending chunks from root  to new array

    # recveiving summation from workers and adding to master sum
    for i in range(1, total_worker):
        receive = comm.recv(source=i,tag=3)
        total+=receive

    # dividing sum with vector size to get average
    average =total/n

    # the final data
    print("Average of Vector 1 : ", average)
    # total time
    print("Execution Time :", MPI.Wtime() - start_time)  # total time takes to task complition

else: # receiving array parts to sum

    v1 = comm.recv(source=root,tag=1)
    result=0
    for i in range(small_slice):
        result += v1[i]

    comm.send(result, dest=root,tag=3)


