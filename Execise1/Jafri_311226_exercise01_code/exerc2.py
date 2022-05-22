import numpy as np
from mpi4py import MPI
np.set_printoptions(threshold=10)
comm = MPI.COMM_WORLD

total_worker = comm.Get_size()
worker_id = comm.Get_rank()
start_time = MPI.Wtime()  # time variable to find time at the end

n = 10**1  # size of array

root = 0  # master with id=0
small_slice = n // total_worker  # slice size
if worker_id == root:  # Condition of checking master to distribute data

    A = np.array(np.random.randint(10, size=[n, n]))
    b = np.array(np.random.randint(10, size=[n, 1]))

    print("Matrix A: ", A,"Matrix Size, N x N: ", A.shape)
    print("Vector b: ", b,"Vector Size, N x 1: ", b.shape)


    # dividing matrix and vector into equal parts and sending each part to worker for processing
    for i in range(1, total_worker):  # for loop to send data

        start_index = (i - 1) * small_slice
        end_index = i * small_slice

        sliceA = A[start_index:end_index, :]
        sliceB = b[start_index:end_index]

        comm.send(sliceA, dest=i, tag=1)
        comm.send(sliceB, dest=i, tag=2)
        # sending the chunks of array to every worker

    resultVector = []
    # Master calculation of its slice
    start_index = (n - small_slice)
    end_index = n

    sliceA = A[start_index:end_index, :]
    sliceB = b[start_index:end_index]

    last_slice = np.zeros(small_slice, dtype=int)

    # matrix multiplication for slice
    for i in range(small_slice):  # for selecting rows of first matrix
        for k in range(small_slice):  # for multiplying row to column

            last_slice[i] += sliceA[i][k] * sliceB[k]



    for i in range(1, total_worker):
        receive = comm.recv(source=i, tag=3)
        resultVector.append(receive)

    resultVector.append(last_slice)

    # the final data
    print("Matrix Vector multiplication result, c  : ", np.array(resultVector).reshape(-1))
    print("Output Size, N x 1: ", b.shape)
    # total time
    print("Execution Time :", MPI.Wtime() - start_time)  # total Execution Time to task complition

else:  # receiving array parts to sum

    sliceA = comm.recv(source=root, tag=1)
    sliceB = comm.recv(source=root, tag=2)
    result = np.zeros(small_slice, dtype=int)

    # matrix multiplication for slice
    for i in range(small_slice):  # for selecting rows of first matrix
        for k in range(small_slice):  # for multiplying row to column
            result[i] += sliceA[i][k] * sliceB[k]

    comm.send(result, dest=root, tag=3)