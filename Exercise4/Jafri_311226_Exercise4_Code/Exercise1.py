from cProfile import label
import math
import os
import re
from turtle import color
import matplotlib.pyplot as plt
from mpi4py import MPI
import string
import numpy as np
import random
import sys
random.seed(0)
np.printoptions(threshold=10)


comm = MPI.COMM_WORLD # setting up the MPI communicator
total_worker = comm.Get_size() # getting number of workers
rank = comm.Get_rank() # storing rank of each worker
start_time = MPI.Wtime()  # time variable to find time at the end
master = 0  # master with id=0
epoch = int(sys.argv[1] )# number of epoch
numberOffeatures = 482 # total number of features in VirusShare Dataset
µ = 10**(-9) # setting value of learning rate

def getData(dir):
    '''Reading files absoulute path from dataset directory and 
    reading each file row and storing it as record in a data list'''
    data = []
    for dir_path, _, filenames in os.walk(dir):
        if len(filenames) > 0:
            for file in filenames:
                # for complete refrence of filepath
                abs_path_train = os.path.abspath(os.path.join(dir_path, file))  
                
                with open(abs_path_train, 'r',encoding='utf-8') as f:
                    Lines = f.readlines()
                    
                    for line in Lines:
                        #removing linespace and adding to list
                        data.append(line.strip('\n'))
        
    return data
def getFeaturesAndTargets(data):
    ''' Reading each line of VirusShare dataset and extracting Targets Y 
    which are first element of every row and Features X which
    position and values are separated by colon ':' '''
    
    Y = np.zeros((len(data),1),dtype=np.float16)
    X = np.zeros((len(data),numberOffeatures),dtype=np.int16)

    row = 0
    for d in data:
        columns = d.split(' ')
        
        # extracting Y targets for every row in data
        Y[row] = float(columns[0])
        
        # extracting X features for every row in data
        col = 0
        for i in range(1,len(columns)):
            if columns[i]!= '':
                featureIndex = int(columns[i].split(':')[0])  
                featureValue = int(columns[i].split(':')[1])
                X[row][featureIndex] = featureValue
        row += 1

    return X,Y

def calculateChunks(data):
    # Calculating chunk size for each worker
    chunk_size = len(data) // total_worker
    
    chunkList = []
    for i in range(1,total_worker+1 ): 
        
        if i== total_worker:
            # for not equaly divided dataset rows, for last chunk adding all remaing items 
            chunkList.append(data[(i - 1) * chunk_size:])
        else:
            chunkList.append(data[(i - 1) * chunk_size:i * chunk_size])
    
    return chunkList

def squaredLoss(X,Y,B):
    ''' Computes squared loss between prediction and actual targets Y'''
    
    return np.sum((Y - np.dot( X, B.T))**2)

def lossGrad(X,Y,B):
    '''Computes gradient of Loss function with respect to paramtere B for 1 instance of SGD'''
    
    return (-2) * ( Y.item() - np.dot( X, B.T).item() ) * X

    
def stochasticGradDecent(dataTrain,µ,B):
    '''Computes Stochastic Gradient Decent step for 
    one chunk of Training data and return updated Parameter B'''
    
    B_new =np.empty(B.shape)
    
    for j in range(len(dataTrain)):
        B_new =  B - (µ * lossGrad(dataTrain[j,:-1].reshape(1,numberOffeatures+1) , dataTrain[j,-1:] , B) )

    return B_new

if rank == master:
    # reading dataset from directory and storing it in data list
    data = getData('Dataset')
    
    # extracting features X and targets Y
    X,Y = getFeaturesAndTargets(data)
    
    # adding Bias column B0 of ones in X
    bias_column = np.ones( shape=(len(X) , 1))
    X = np.append(bias_column,X,axis=1)

    # appending X and Y toghether so that index of X and its target Y remains same after shuffling
    data = np.append(X,Y,axis=1)

    # splitting data into training and test splits with 80:20 ratio
    dataTrain = data[0 : math.floor(len(data)*0.8), : ]
    dataTest = data[math.floor(len(data)*0.8) + 1 :, : ]
    
    # initializing param B with zeros
    B = np.zeros((1,numberOffeatures+1))
    
    # storing length of training data for RMSE calculation
    lenTrain = len(dataTrain)
else:
    B=None
    

trainLoss = []
testLoss = []
iteration=0

# broadcasting param B to all workers
B = comm.bcast(B, root=0)

while iteration < epoch :
    iteration+=1
    
    if rank == master:
        chunkListTrain = calculateChunks(dataTrain)
        chunkListTest = calculateChunks(dataTest)

    else:
        chunkListTrain = None
        chunkListTest = None

    # scattering data chunks to all workers 
    dataTrain = comm.scatter(chunkListTrain,root=0)
    dataTest = comm.scatter(chunkListTest,root=0)

    # shuffling dataset 
    np.random.shuffle(dataTrain)
    
    # computing updated param B after SGD step 
    B = stochasticGradDecent (dataTrain,µ,B) 
    
    # calculating squared loss for trainin data for each worker    
    traningsquareloss = squaredLoss(dataTrain[:,:-1].reshape(len(dataTrain),numberOffeatures+1),dataTrain[:,-1:],B)
    totalTraningSquaredLoss= np.zeros(1)
    
    # adding squared loss from each workers at master
    comm.Reduce(traningsquareloss,totalTraningSquaredLoss,op=MPI.SUM, root=0)
    
    # calculating squared loss for trainin data for each worker    
    testgsquareloss = squaredLoss(dataTest[:,:-1].reshape(len(dataTest),numberOffeatures+1),dataTest[:,-1:],B)
    totalTestSquaredLoss= np.zeros(1)
    
    if total_worker>1:
        # adding squared loss from each workers at master
        comm.Reduce(testgsquareloss,totalTestSquaredLoss,op=MPI.SUM, root=0)
    else:
        totalTestSquaredLoss = testgsquareloss

    if total_worker>1:
        # computing sum of param B from each worker at master
        B_avg = np.empty((1,numberOffeatures+1))
        comm.Reduce(B,B_avg,op=MPI.SUM, root=0)
    else:
        B_avg=B

    
    if rank == master:
        # computed train rmse and appending to list for plotting for each epoch
        trainLoss.append(np.sqrt(totalTraningSquaredLoss/lenTrain))
        
        # computed test rmse and appending to list for plotting for each epoch
        testLoss.append(np.sqrt(totalTestSquaredLoss/lenTrain))
        
        # computing average of param B from each worker by dividing with total number of workers at master
        B = B_avg/total_worker
    else:
        B = None
    # broadcasting Param B to each worker for next epoch
    B = comm.bcast(B, root=0)

 

    
if rank == 0:
    print('******************** Experiment *************************')
    print('Total Number of Workers : ',  total_worker )
    print('Train Loss : ', trainLoss )
    print('Test Loss : ', trainLoss )
    print('Total Epochs',epoch)
    totalTime = MPI.Wtime() - start_time
    print('Total Execution Time : ',  MPI.Wtime() - start_time )
    mode = 'w'  if total_worker==1 else 'a'
    with open('Timefor'+str(epoch)+'epochs.txt', mode) as f:
        f.write(str(total_worker) +','+str(totalTime) + '\n')
        
    ## plotting train and test loss convergence    
    # plt.plot(trainLoss,label = 'Train')
    # plt.subplot(1, 2, 1)
    # plt.title('PSGD Convergence Plot for '+str(total_worker) +' Workers')
    # plt.xlabel('Epochs')
    # plt.ylabel('RMSE')
    # plt.legend()
    # plt.rcParams["figure.figsize"] = (25,25)

    
    
    # For Standard Deviation = 0.1


    # For Standard Deviation = 1
    plt.subplot(1, 2, 1)
    plt.plot(trainLoss,label = 'Train',color ='slateblue')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title("Squential SGD convergence for Training")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(testLoss,label = 'Test',color ='orange')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.title("Squential SGD convergence for Test")
    plt.show()
    #plt.savefig('PSGD Convergence Plot for '+str(total_worker) +' Workers,' +str(epoch)+'Epochs'+ '.png')

# print(trainLoss[0:10])