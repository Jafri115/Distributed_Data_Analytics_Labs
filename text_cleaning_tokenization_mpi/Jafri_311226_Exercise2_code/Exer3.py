import json
import math
from collections import Counter
from mpi4py import MPI

comm = MPI.COMM_WORLD  # setting up the MPI communicator
total_worker = comm.Get_size()  # getting number of workers
rank = comm.Get_rank()  # storing rank of each worker
start_time = MPI.Wtime()  # time variable to find time at the end
master = 0  # master with id=0


def calculateSumTF(tfDictChunk):
    '''
    Calculating sum of all term frequency for all term in a document

    :param tfDictChunk: List of Term frequency dictionary for each document
    :return:  dictionary of every term frequency across all documents
    '''
    idfDictChunk = {}

    for key in tfDictChunk:
        docDic = tfDictChunk[key]
        for wordKey in docDic:
            if (idfDictChunk.get(wordKey) != None):  # if token already exist, add 1 in token
                idfDictChunk.update({wordKey: idfDictChunk.get(wordKey) + 1})
            else:  # if token does not exist put 1
                idfDictChunk.update({wordKey: 1})
    return idfDictChunk


if rank == master:  # Condition of checking master to distribute data
    # reading term frequency dictionary from last exercise
    tfDict = json.load(open("TF.txt"))

    C = len(tfDict)  # total number of documents

    chunk_size = len(tfDict) // total_worker  # each chunk size processed by worker

    for i in range(1, total_worker):  # for loop to send data

        # slicing dictionary for sending to workers
        tfDictChunk = {'document' + str(k): tfDict['document' + str(k)] for k in
                       range(((i - 1) * chunk_size) + 1, (i * chunk_size) + 1)
                       if 'document' + str(k) in tfDict}
        # sending to other workers
        comm.send(tfDictChunk, dest=i, tag=1)

    ## master calculation of own chunk
    tfDictChunk = {'document' + str(k): tfDict['document' + str(k)] for k in
                   range((i * chunk_size) + 1, len(tfDict) + 1) if
                   'document' + str(k) in tfDict}
    idfDict = Counter(calculateSumTF(tfDictChunk))
    '''receiving chunks from workers and concatinating them to make one dictionary 
    in which each term tell in how many documents it appears'''
    for i in range(1, total_worker):
        idfDictChunk = comm.recv(source=i, tag=3)
        idfDict += Counter(idfDictChunk)

    # diving C with each term in this dictionary and taking log to give final Idf Dictionary
    finalIdfDict = {}
    for key in dict(idfDict):
        finalIdfDict.update({key: math.log(C / idfDict[key])})

    # writing the IDF dictionary to a file for later exercises
    json.dump(finalIdfDict, open("IDF.txt", 'w'))

    print("Idf for terms starts with ast: ", {k:v for k,v in finalIdfDict.items() if k.startswith('ast')})
    print("Execution Time :", MPI.Wtime() - start_time)  # total time takes to task completion
else:

    # receiving the chunk of term frequency dictionary list from master
    tfDictChunk = comm.recv(source=master, tag=1)

    comm.send(calculateSumTF(tfDictChunk), dest=master, tag=3)
