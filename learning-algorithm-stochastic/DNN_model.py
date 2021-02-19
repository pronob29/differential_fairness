# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:00:04 2020

@author: islam
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from DF_Training import stochasticCountModel, computeBatchCounts, fairness_loss, sf_loss, prule_loss

from fairness_metrics import computeEDFforData
# fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden1,hidden2,hidden3,output_size):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden2,hidden3)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.outputLayer = nn.Linear(hidden3,output_size)
        self.out_act = nn.Sigmoid() 
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop3(out)
        out = self.outputLayer(out)
        out = self.out_act(out)
        return out
#%% train the model without fairness constraint
def training_typical(input_size,hidden1,hidden2,hidden3,output_size,learning_rate,num_epochs,trainData,trainLabel,miniBatch):
    dnn_model = NeuralNet(input_size,hidden1,hidden2,hidden3,output_size)
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(dnn_model.parameters(),lr = learning_rate)
    #optimizer = optim.Adadelta(modelFair.parameters())
    #optimizer = optim.Adamax(modelFair.parameters(),lr = learning_rate)
    # Train the netwok
    for epoch in range(num_epochs):
        for batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):
            trainY_batch = trainLabel[batch:(batch+miniBatch)]
            trainX_batch = trainData[batch:(batch+miniBatch)]
            
            trainX_batch = Variable(trainX_batch.float())
            trainY_batch = Variable(trainY_batch.float())
            
            # forward + backward + optimize
            outputs = dnn_model(trainX_batch)
            tot_loss = criterion(outputs, trainY_batch)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step() 
        print('epoch: ', epoch, 'out of: ',num_epochs, 'average loss: ',tot_loss.item())
    return dnn_model 

#%% train the model with differential fairness constraint
def training_fair_model(input_size,hidden1,hidden2,hidden3,output_size,learning_rate,num_epochs,trainData,trainLabel,miniBatch,S,intersectionalGroups,burnIn,stepSize,epsilonBase,lamda):
    
    VB_CountModel = stochasticCountModel(len(intersectionalGroups),len(trainData),miniBatch)
    
    modelFair = NeuralNet(input_size,hidden1,hidden2,hidden3,output_size)
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(modelFair.parameters(),lr = learning_rate)
    #optimizer = optim.Adadelta(modelFair.parameters())
    #optimizer = optim.Adamax(modelFair.parameters(),lr = learning_rate)
    # Train the netwok
    for epoch in range(burnIn):
        for batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):
            trainS_batch = S[batch:(batch+miniBatch)] # protected attributes in the mini-batch
            trainY_batch = trainLabel[batch:(batch+miniBatch)]
            trainX_batch = trainData[batch:(batch+miniBatch)]
            
            trainX_batch = Variable(trainX_batch.float())
            trainY_batch = Variable(trainY_batch.float())

            # forward + backward + optimize
            outputs = modelFair(trainX_batch)
            tot_loss = criterion(outputs, trainY_batch)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
        print('burn-in epoch: ', epoch, 'out of: ',burnIn, 'average loss: ',tot_loss.item())   
    for epoch in range(num_epochs):
        for batch in range(0,np.int64(np.floor(len(trainData)/miniBatch))*miniBatch,miniBatch):
            trainS_batch = S[batch:(batch+miniBatch)] # protected attributes in the mini-batch
            trainY_batch = trainLabel[batch:(batch+miniBatch)]
            trainX_batch = trainData[batch:(batch+miniBatch)]
            
            trainX_batch = Variable(trainX_batch.float())
            trainY_batch = Variable(trainY_batch.float())
            
            VB_CountModel.countClass_hat.detach_()
            VB_CountModel.countTotal_hat.detach_()
            # forward + backward + optimize
            outputs = modelFair(trainX_batch)
            loss = criterion(outputs, trainY_batch)

            # update Count model 
            countClass, countTotal = computeBatchCounts(trainS_batch,intersectionalGroups,outputs)
            #thetaModel(stepSize,theta_batch)
            VB_CountModel(stepSize,countClass, countTotal,len(trainData),miniBatch)
            
            # fairness constraint 
            lossDF = fairness_loss(epsilonBase,VB_CountModel)            
            tot_loss = loss+lamda*lossDF
            
            # zero the parameter gradients
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step() 
            
        print('epoch: ', epoch, 'out of: ',num_epochs, 'average loss: ',tot_loss.item())
    return modelFair
