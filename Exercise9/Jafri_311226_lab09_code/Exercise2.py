from nltk.corpus import stopwords
from mpi4py import MPI
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import dropout, optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader,Dataset  # Gives easier dataset managment by creating mini batches etc.
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
torch.cuda.empty_cache()
import torch.nn.functional as F
import numpy as np
from timeit import default_timer as timer
import torch.multiprocessing as mp
import time
import sys


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_of_classes=10,dropout=0.5):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(
            in_channels = input_channels,
            out_channels = 16,
            kernel_size = (5,5),
            stride=(1,1),
            padding=(0,0)
            
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = (5,5),
            stride=(1,1),
            padding=(0,0)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout1 = nn.Dropout(self.dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.FC1 = nn.Linear(32* 2* 2,512)
        self.FC2 = nn.Linear(512,256)
        self.Output = nn.Linear(256,num_of_classes)
       
  
    def forward(self,x):
  
        x = self.conv1(x)
        x = self.relu(self.pool1(x)) 
        x = self.conv2(x)
        x = self.relu(self.pool2(x))
        x = self.dropout1(x)
        x = self.pool3(x)
        x = x.view(-1, 32* 2* 2)  
        
        x = self.relu(self.FC1(x))
        x = self.FC2(x)
        
        x = self.softmax(self.Output(x))
        
        return x
def accuracy(outputs, labels):
    ''' Calculating accuray of prediction'''

    _, preds = torch.max(outputs, dim = 1) # class with highest probablity is the predicted output
    return(torch.tensor(torch.sum(preds == labels).item()/ len(preds)))   

def trainModel(model,learning_rate,criterion,train_loader,num_epochs,rank):
    # Initializing Optimizer
    optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        lossSum = 0
        accSum = 0

        # iterating through all batches
        for i,sample_batched in enumerate(train_loader):

            # setting gradient to zero before forward pass
            optimizer.zero_grad()

            # model predict
            yhat=model(sample_batched[0])

            # model loss
            loss=criterion(yhat.squeeze(),sample_batched[1].squeeze())
            lossSum += loss.item()

            # model accuracy
            acc = accuracy(yhat.squeeze(),sample_batched[1].squeeze())
            accSum += acc.item()

            # calculating gradient through backprop
            loss.backward(retain_graph=True)

            # updating parameters
            optimizer.step()
                    # writing accuracy and loss to tensorboard logs
        lossAvg = lossSum /len(train_loader)
        accAvg = accSum/len(train_loader)

        
    print (f'Worker: [{rank}],Train Loss: {lossAvg:.4f},Train Accuracy: {accAvg:.4f}')

    print('Finished Training at rank:',rank)
    return model
def evalModel(model,criterion,test_loader):
    accSum = 0
    lossSum = 0

    # for evaluating model on test data with no gradient calculation
    with torch.no_grad():
        for i,sample_batched in enumerate(test_loader):

            # model predict
            yhat=model(sample_batched[0])

            # model loss
            loss=criterion(yhat.squeeze(),sample_batched[1].squeeze())
            lossSum += loss
            # model acc
            acc = accuracy(yhat.squeeze(),sample_batched[1].squeeze())
            accSum += acc

            # writing accuracy and loss to tensorboard logs
    
    totalloss = lossSum/len(test_loader)
    totalAccuracy = accSum/len(test_loader)
    print('\nTest Data: Average loss: {:.4f}, Final Accuracy: ({:.0f}%)\n'.format(totalloss,100. * totalAccuracy))

    
    return totalAccuracy.item(),totalloss.item()


if __name__ == '__main__':
    # setting device
    device =  "cpu"
    torch.device(device)

    in_channels = 1 # input channels of image
    num_classes = 10 # number of output class [0-10]
    num_epochs = 30
    batch_size= 256
    transform = transforms.Compose([transforms.Resize((32,32)),
    transforms.ToTensor()
    ])

    # getting training and test data
    train_dataset = datasets.MNIST(root="MNIST/", train=True, transform=transform)
    test_dataset = datasets.MNIST(root="MNIST/", train=False, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size ,shuffle=False)

    # different setting of learning rates
    learning_rate =0.001

    criterion = nn.NLLLoss()
        
    # Initialize network
    model = CNN(input_channels=in_channels, num_of_classes=num_classes,dropout=0.4).to(device)


    
    # initialize model 
    mp.set_start_method('spawn')
    model = CNN(input_channels=in_channels, num_of_classes=num_classes,dropout=0.4).to(device)
    model.share_memory()
    num_processes = 5
    processes = []
    start_time = time.perf_counter()
    print('-----------------------------------------------------------------')
    print('|--------------Experiment: Number of processes='+str(num_processes)+' ---------------|')
    print('-----------------------------------------------------------------')
    for rank in range(num_processes):
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_processes, rank=rank) 
        train_loader = DataLoader(dataset=train_dataset,sampler=train_sampler,batch_size=batch_size)
        ## initializing and starting p processes 
        p = mp.Process(target=trainModel, args=(model, learning_rate,criterion, train_loader, num_epochs,rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    total_time = time.perf_counter() - start_time
    
    # calculating time for training        
    total_time = MPI.Wtime() - start_time
    print('Total time for training: {}'.format(total_time))

    
    # evaluating model on test data
    testAcc,testLoss = evalModel(model,criterion,test_loader)
    # storing time and accuracy for speedup and accuracy graphs
    mode = 'w'  if num_processes==1 else 'a'
    with open('TimeforTraining2.txt', mode) as f:
        f.write(str(num_processes) +','+str(total_time)+','+str(testAcc*100) + '\n')