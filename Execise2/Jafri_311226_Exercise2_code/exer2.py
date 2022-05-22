import json
from collections import Counter
from mpi4py import MPI

comm = MPI.COMM_WORLD  # setting up the MPI communicator
total_worker = comm.Get_size()  # getting number of workers
rank = comm.Get_rank()  # storing rank of each worker
start_time = MPI.Wtime()  # time variable to find time at the end
master = 0  # master with id=0


def calculate_Tf(tokenizeDocChunk, docIds):
    '''
    Calculating Term Frequency for each word in the document list by using counter
    Normalizing by dividing total number of terms in the document

    :param tokenizeDocChunk: Chunk of Tokenized Data list
    :param docIds: Chunk of document ids list
    :return: Term frequency of each word in the document
    '''
    tfDictChunk = {}
    i = 0 # for iterating document ids

    for lines in tokenizeDocChunk:
        words = dict(Counter(lines)) # counting term frequency for each word
        numOftokens = sum(words.values()) # total terms in a document
        for key in words:
            words.update({key: words[key] / numOftokens}) # updating normalize term frequency
        tfDictChunk.update({'document' + str(docIds[i]): words})
        i += 1

    return tfDictChunk


if rank == master:  # Condition of checking master to distribute data
    # reading tokenized data text file from last exercise
    with open('TokenizeData.txt', 'r') as f:
        # removing new line and commas and splitting each line with comma(,) into a list
        tokenizeDoc = [(line.rstrip('\n')).replace("'", "").replace(' ', '').split(",") for line in f]

    docIds = range(1, len(tokenizeDoc) + 1)  # for document ids stored as key in dictionary
    chunk_size = len(tokenizeDoc) // total_worker  # each chunk size processed by worker

    for i in range(1, total_worker):  # for loop to send data
        # sending the chunk of tokenized data list to workers
        comm.send(tokenizeDoc[(i - 1) * chunk_size:i * chunk_size], dest=i, tag=1)
        # sending the document id for keys in dictionary
        comm.send(docIds[(i - 1) * chunk_size:i * chunk_size], dest=i, tag=2)

    # master calculation of own chunk
    docFreqDict = calculate_Tf(tokenizeDoc[i * chunk_size:], docIds[i * chunk_size:])

    # receiving chunks of term frequencies from other workers and concatenating them
    for i in range(1, total_worker):
        docFreqDictChunk = comm.recv(source=i, tag=3)
        docFreqDict.update(docFreqDictChunk)

    # ordering the term frequency dictionary with key as document ids
    orderDocFreq = dict(sorted(docFreqDict.items()))

    # writing the TF dictionary to a file for later exercises
    json.dump(orderDocFreq, open("TF.txt", 'w'))

    print("Term Frequencies : document1", orderDocFreq['document1'])
    print("Term Frequencies : document19997", orderDocFreq['document19997'])
    print("Execution Time :", MPI.Wtime() - start_time)  # total time takes to task complition

else:
    # receiving the chunk of tokenized data list from master
    tokenizeDocChunk = comm.recv(source=master, tag=1)

    # receiving the document ids for keys in dictionary
    docIds = comm.recv(source=master, tag=2)

    comm.send(calculate_Tf(tokenizeDocChunk, docIds), dest=master, tag=3)
