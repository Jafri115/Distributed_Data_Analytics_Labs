import os
from nltk.corpus import stopwords
import nltk
import re
from mpi4py import MPI
import string

comm = MPI.COMM_WORLD # setting up the MPI communicator
total_worker = comm.Get_size() # getting number of workers
rank = comm.Get_rank() # storing rank of each worker
start_time = MPI.Wtime()  # time variable to find time at the end
master = 0  # master with id=0


def cleanTokenizeData(filesInChunk):
    '''Opening files in the Chunk and removing number ,english stopwords,punctuations and
    words smaller than 2 characters from text and then tokenizing the documents
    Each row of return list represent on document'''
    docList = []
    for file in filesInChunk:
        f = open(file, "r")
        docList.append(f.read())

    cleanData = [(lambda x: [word for word in
                             nltk.word_tokenize(re.sub(r'[0-9]', '', x).lower().strip(string.punctuation)) if
                             word.isalnum() and word not in stopwords.words('english') and len(word) > 2])(doc) for doc
                 in docList]
    return cleanData


if rank == master:  # Condition of checking master to distribute data

    def getDocumentList(dir):
        '''Reading files absoulute path from dataset directory and returning it in a list '''
        file_paths = []
        for dir_path, _, filenames in os.walk(dir):
            if len(filenames) > 0:
                for file in filenames:
                    abs_path_train = os.path.abspath(os.path.join(dir_path, file))  # for complete refrence of filepath
                    file_paths.append(abs_path_train)
        return file_paths


    fileList = getDocumentList('20_newsgroups')

    chunk_size = len(fileList) // total_worker

    # Splitting file path list between all workers
    for i in range(1, total_worker):  # for loop to send data
        filesInChunk = fileList[(i - 1) * chunk_size:i * chunk_size]
        comm.send(filesInChunk, dest=i, tag=1)

    # master calculation cleaning the chunk
    tokenDocList = cleanTokenizeData(fileList[i * chunk_size:])

    for i in range(1, total_worker):
        tokenData = comm.recv(source=i, tag=3)
        tokenDocList += tokenData

    print('Tokenized and clean Data List:', tokenDocList)
    print('Execution time :', MPI.Wtime() - start_time)

    '''Writing the tokenize data list to a file to be used in next exercise.
    Each Document is separated by new line'''
    with open('TokenizeData.txt', 'w') as f:
        for tokendoc in tokenDocList:
            f.write(str(tokendoc)[1: -1] + '\n') # removing brackets before writing


else:  # workers calculations
    filesInChunk = comm.recv(source=master, tag=1)
    comm.send(cleanTokenizeData(filesInChunk), dest=master, tag=3)
