import random
from mpi4py import MPI
import numpy as np
import json
import math
from collections import Counter
from mpi4py import MPI
import numpy as np
np.set_printoptions(threshold=10000000)
np.random.seed(0)

comm = MPI.COMM_WORLD  # setting up the MPI communicator
total_worker = comm.Get_size()  # getting number of workers
rank = comm.Get_rank()  # storing rank of each worker
start_time = MPI.Wtime()  # time variable to find time at the end
master = 0  # master with id=0
converge = 0 # for checking convergence condition
iteration = 0 # to run epoxs
max_iter = 500 # total number of iteration
threshold = 10**-5
def randomCentroid(data, k):
    randomRows = np.random.randint(data.shape[0], k)
    centroid = data[randomRows]

    return centroid

def initailizeRandomCentroid(tfIdfDict, cleanIdfDict,k):
    tfIdfDictKeys = random.sample(list(tfIdfDict),k)
    tfIdfDictRand = {k:tfIdfDict[k] for k in tfIdfDictKeys}
    centroid = transformData(tfIdfDictRand,cleanIdfDict)

    return centroid


def euclidean_distance(dataPoint, centroid):
    '''returns euclidean distance between two instances'''
    return np.sqrt(np.sum((np.array(dataPoint) - np.array(centroid)) ** 2))
    

def calculateDistanceMatrix(tfIdfVectorsChunk, centroids):
    '''returns distance matrix which contains distances of each points from each centroids,
    Every row of distance matrix represents an instance and each columns represents centroids'''

    rows = tfIdfVectorsChunk.shape[0]
    k = centroids.shape[0]
    distanceMatrix = np.zeros((rows, k))
    for r, record in enumerate(tfIdfVectorsChunk):
        for c, centroid in enumerate(centroids):
            distanceMatrix[r][c] = euclidean_distance(record, centroid) 
    return distanceMatrix

def assignMembership (distanceMatrix):
    '''In distance matrix, for every row instance get minimum value in that row which tells which is closet centroid.
    Assign that column index as cluster of that row.Return this matrix of assignments'''
    rows= distanceMatrix.shape[0]
    membershipVector = []
    for i in range(rows):
            membershipVector.append([distanceMatrix[i].argmin()])

    return np.array(membershipVector)


def transformData(tfIdfDict, cleanIdfDict):
    '''
    Transforn tf-idf dictionary to vector form. where every row represents on document and 
    there is column represent on token word in the corpus. Each entry in this matrix is tf-idf of 
    that document for a perticular word.

    '''
    cols = len(cleanIdfDict)  # tokens as M features
    rows = len(tfIdfDict)  # number of Documents as N rows in dataset

    tf_idf_mat = np.zeros((rows, cols))
    r = 0  # for iterating over rows
    for doc in tfIdfDict:
        c = 0 # for iterating over columns
        for token in cleanIdfDict:
              
            # if there is a token in doc, then place tf-idf of token else zero value will remain
            if (tfIdfDict[doc].get(token) != None):
                tf_idf_mat[r][c] = tfIdfDict[doc].get(token)

            c += 1

        r += 1


    return tf_idf_mat


if rank == master:  # Condition of checking master to distribute data
    # reading tf-idf scores stored in last exercise
    tfIdfDict = dict((json.load(open("tf_Idf.txt")).items()))

    # reading idf files so that we know how many terms are there in total
    # and these are then converted into feature
    # columns M
    idfDict = dict((json.load(open("IDF.txt")).items()))

    print('Number of total Documents:',len(tfIdfDict))


    # removing tokens with idf=0 because their tf-idf will be 0 and wont help in clustering.
    cleanIdfDict = {key: val for key, val in idfDict.items() if val != 0}
    print('Number of total Words:',len(cleanIdfDict))

    chunk_size = len(tfIdfDict) // total_worker  # each chunk size processed by worker

    k = 2
    # initializing centroid randomly 
    #centroids = np.random.uniform(0.5,1,size=(k,len(cleanIdfDict))) 
    centroids = initailizeRandomCentroid(tfIdfDict,cleanIdfDict,k)
    
    # splitting tf-idf dictionary into chunks 
    tfIdfDictChunkList = []
    for i in range(1, total_worker+1):  # for loop to send data
        # slicing tf dictionary for sending to workers
        tfIdfDictChunk = {'document' + str(k): tfIdfDict['document' + str(k)] for k in
                          range(((i - 1) * chunk_size) + 1, (i * chunk_size) + 1)
                          if 'document' + str(k) in tfIdfDict}
        if(i==total_worker):
            tfIdfDictChunk = {'document' + str(k): tfIdfDict['document' + str(k)] for k in
                    range(((i - 1) * chunk_size) + 1, len(tfIdfDict) + 1)
                    if 'document' + str(k) in tfIdfDict}

        tfIdfDictChunkList.append(tfIdfDictChunk)

    lossList =[]
else:
    tfIdfDictChunkList = None
    cleanIdfDict = None
    tran_tf_idfs = None
    centroids = None
    
    
# everyprocess recieves one chunk  tf-idf dictionary
tfIdfDictChunk = comm.scatter(tfIdfDictChunkList,root=0)

# broadcasting Idf dictionary for column features
idfDict = comm.bcast(cleanIdfDict, root=0)

# transforming dictionary chunk into vector
tfIdfVectorsChunk = transformData(tfIdfDictChunk, idfDict)

# broadcasting initial centroids
centroids = comm.bcast(centroids, root=0)



while (iteration < max_iter and converge == 0):
    
    new_Centroids = centroids.copy()

    # distance matrix of shape(numberOfPoints,number of columns) where each rows tell the distance from each centroid
    distanceMatrix = calculateDistanceMatrix(tfIdfVectorsChunk, new_Centroids)
    
    # membership vector of shape(numberOfPoints,1) where each row tells which is closest centroid
    memberShipVectorChunk = assignMembership(distanceMatrix)
    totalloss= np.zeros(1)
    loss = 0
    # for i in range(len(memberShipVectorChunk)):
    #     loss += euclidean_distance(tfIdfVectorsChunk[i],new_Centroids[memberShipVectorChunk[i]])
    # comm.Reduce(loss,totalloss,op=MPI.SUM, root=0)
    numberOfClusters , features = centroids.shape

    '''# for each cluster, finding member points index and getting those points 
    and adding each feature values of those member points and 
    dividing with total number of members to get the local mean of that cluster'''
    for i in range(numberOfClusters):
        
        centroidKIndex = np.where(memberShipVectorChunk== i) # gives indexes of cluster member

        if len(np.array(centroidKIndex)[0]) !=0:
            new_Centroids[i]= np.sum(tfIdfVectorsChunk[np.array(centroidKIndex)[0],:], axis=0,keepdims=True)/len(np.array(centroidKIndex)[0]) 

    # gathering local means at master worker
    localMeansList = comm.gather(new_Centroids, root=0)
    
    if rank == master:

        # calculating globle centroid by taking mean of local centroids returns by each worker
        updatedCentroids = np.empty([numberOfClusters,features])
        centroidDistance = 0
        for k in range(numberOfClusters):
            i = 0
            for localMean in localMeansList:
                if i == 0:
                    globleMeanNp = localMean[k,:].reshape(1,features)
                else:
                    globleMeanNp = np.concatenate((globleMeanNp,  localMean[k,:].reshape(1,features)), axis=0)
                    
                i+=1
        
            updatedCentroids[k] = np.sum(globleMeanNp, axis=0,keepdims=True)/(total_worker)

            # calculating distance between new centroids and last iteration centroids
            centroidDistance += euclidean_distance(updatedCentroids[k],centroids[k])
     
        if(centroidDistance <threshold): # checking if centroids are not moving, then it means it is converged and comming out of loop
            converge = 1
        print('------------Iteration '+str(iteration)+'---------')
        print('Distance Btw new and old centroids:',centroidDistance)
        print('totalLoss:',totalloss)
        lossList.append(totalloss[0])
    else:
        updatedCentroids = None
        centroids = None
    # updating centroids    
    centroids = updatedCentroids
    # broadcasting new centroids 
    centroids = comm.bcast(centroids, root=0)
    iteration = iteration + 1
    # broadcasting convergence variable so that all workers come out of loop
    converge = comm.bcast(converge, root=0)
    
 
if rank == 0:
    
    clusteringTime =   MPI.Wtime()-start_time   
    print('Total Time :', clusteringTime)
    mode = 'w'  if total_worker==1 else 'a'
    if(total_worker==1):
        with open('loss.txt', 'a') as f:
            f.write(",".join(str(item) for item in lossList) + '\n')
    with open('Timefork'+str(numberOfClusters)+'.txt', mode) as f:
        f.write(str(total_worker) +','+str(clusteringTime) + '\n')