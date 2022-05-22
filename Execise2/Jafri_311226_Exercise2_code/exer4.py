import json
from mpi4py import MPI

comm = MPI.COMM_WORLD  # setting up the MPI communicator
total_worker = comm.Get_size()  # getting number of workers
rank = comm.Get_rank()  # storing rank of each worker
start_time = MPI.Wtime()  # time variable to find time at the end
master = 0  # master with id=0


def calculateTfIdf(tfDictChunk, idfDic):
    ''' Calculating tf-idf by multiplying tf of each term in document with idf of that term'''
    tf_idfDictChunk = {}
    for key in tfDictChunk:
        docDic = tfDictChunk[key]
        tdfIdfDoc = {}
        for wordKey in docDic:
            if (idfDic.get(wordKey) != None):
                tdfIdfDoc.update({wordKey: docDic.get(wordKey) * idfDic.get(wordKey)})
        tf_idfDictChunk.update({key: tdfIdfDoc})
    return tf_idfDictChunk


if rank == master:  # Condition of checking master to distribute data#

    # reading term frequency dictionary from last exercise
    tfDict = json.load(open("TF.txt"))
    # reading term frequency dictionary from last exercise
    idfDic = json.load(open("IDF.txt"))

    C = len(tfDict)  # total number of documents

    chunk_size = len(tfDict) // total_worker  # each chunk size processed by worker

    for i in range(1, total_worker):  # for loop to send data
        # slicing tf dictionary for sending to workers
        tfDictChunk = {'document' + str(k): tfDict['document' + str(k)] for k in
                       range(((i - 1) * chunk_size) + 1, (i * chunk_size) + 1)
                       if 'document' + str(k) in tfDict}

        comm.send(tfDictChunk, dest=i, tag=1)

        # sending Idf dictionary to all workers
        comm.send(idfDic, dest=i, tag=2)

    ## master calculation of own chunk
    tfDictChunk = {'document' + str(k): tfDict['document' + str(k)] for k in
                   range((i * chunk_size) + 1, len(tfDict) + 1) if
                   'document' + str(k) in tfDict}
    tf_IdfDict = calculateTfIdf(tfDictChunk, idfDic)

    '''receiving chunks from workers and concatinating them to make one dictionary 
    of documents where each value is also a dictionary of Terms idf in that document'''
    for i in range(1, total_worker):
        tf_idfDictChunk = comm.recv(source=i, tag=3)
        tf_IdfDict.update(tf_idfDictChunk)

    print("Tf-Idf : document19997:", tf_IdfDict['document19997'])
    print("length of Tf-Idf Dictionary : ", len(tf_IdfDict))

    print("Execution Time :", MPI.Wtime() - start_time)  # total time takes to task completion

    # writing the tf-IDF dictionary to a file
    json.dump(tf_IdfDict, open("tf_Idf.txt", 'w'))
else:  # receiving array parts to sum
    # receiving the chunk of term frequency dictionary list from master
    tfDictChunk = comm.recv(source=master, tag=1)
    # receiving whole IDF dictionary list from master
    idfDic = comm.recv(source=master, tag=2)

    comm.send(calculateTfIdf(tfDictChunk, idfDic), dest=master, tag=3)
