import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

total_worker = comm.Get_size()
worker_id = comm.Get_rank()
start_time = MPI.Wtime()  # time variable to find time at the end

n = 10 ** 4  # size of array

root = 0  # master with id=0
small_slice = n // total_worker  # slice size
if worker_id == root:  # Condition of checking master to distribute data

    v1 = np.array(np.random.randint(10, size=n))
    v2 = np.array(np.random.randint(10, size=n))

    print("Vector 1: ", v1)
    print("Vector 2: ", v2)
    print("Vector Size:: ", n)

    # slicing the as the number of workers3


    for i in range(1, total_worker):  # for loop to send data

        start_index = (i - 1) * small_slice
        end_index = i * small_slice

        v1_slice = v1[start_index:end_index]
        v2_slice = v2[start_index:end_index]

        comm.send(v1_slice, dest=i, tag=1)
        comm.send(v2_slice, dest=i, tag=2)
        # sending the chunks of array to every worker

    resultVector = []
    # Master calculation of its slice
    start_index = (n - small_slice)
    end_index = n

    v1_slice = v1[start_index:end_index]
    v2_slice = v2[start_index:end_index]

    last_slice = np.zeros(small_slice, dtype=int)
    for i in range(small_slice):
        last_slice[i] = v1_slice[i] + v2_slice[i]  # appending chunks from root  to new array

    for i in range(1, total_worker):
        receive = comm.recv(source=i, tag=3)
        resultVector.append(receive)

    resultVector.append(last_slice)

    # the final data
    print("Summation Result  : ", np.array(resultVector).reshape(-1))
    # total time
    print("Execution Time :", MPI.Wtime() - start_time)  # total Execution Time to task complition

else:  # receiving array parts to sum

    v1 = comm.recv(source=root, tag=1)
    v2 = comm.recv(source=root, tag=2)

    result = np.zeros(small_slice, dtype=int)
    for i in range(small_slice):
        result[i] = v1[i] + v2[i]

    comm.send(result, dest=root, tag=3)