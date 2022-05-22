import numpy as np
from mpi4py import MPI
np.set_printoptions(threshold=10)
comm = MPI.COMM_WORLD

total_worker = comm.Get_size()
rank = comm.Get_rank()
start_time = MPI.Wtime()  # time variable to find time at the end
n = 10**1 # matrix size
root = 0  # master with id=0
small_slice = n // total_worker  # slice size

if rank == root: #master worker
    # initialize matrices
    A = np.array(np.random.randint(10, size=[n, n]))
    B = np.array(np.random.randint(10, size=[n, n]))
    print('Matrix A:', A,"Matrix A Size, N x N: ", A.shape)
    print('Matrix B:', B,"Matrix B Size, N x N: ", B.shape)
    a_chunks =[]

    """
    slicing Matrix A into chunks and appending to
    an array(size must be equal to number of workers)
    so that chunks can be scattered
    """
    for i in range(1, total_worker+1):

        start_index = (i - 1) * small_slice
        end_index = i * small_slice
        if(i==total_worker): #for Matrix sizes not fully divided with workers, remaing part goes to last chunk
            end_index=n
        sliceA = A[start_index:end_index, :]
        a_chunks.append(sliceA)

else:
    a_chunks = None
    B = None
# everyprocess recieves one chunk
recvA = comm.scatter(a_chunks,root=0)

# second matrix is broadcasted because each chunk in Matrix A is multiplied with whole B matrix
B = comm.bcast(B, root=0)
rA,cA = recvA.shape
rB,cB = B.shape


c_Chunks = np.zeros((rA,cB))
for i in range(rA):  # for selecting rows of first matrix
    for j in range(cB):  # for multiplying row to column
        for k in range(rB):
            c_Chunks[i][j] += int(recvA[i][k] * B [k][j])
# Resultant product or chunk of A and B give chunks of C which are gathered at master worker
c = comm.gather(np.array(c_Chunks), root = 0)

if rank == root:

    # concatenating chunks to give final dot product
    C = np.array(c[0])
    for i in range(1,total_worker):
        C = np.concatenate((C, np.array(c[i])), axis=0)
    print('resultant matrix C: ', C.astype(int),"Matrix C Size, N x N: ", C.shape)
    print("Execution Time :", MPI.Wtime() - start_time)  # total Execution Time to task complition