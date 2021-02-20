# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 21:52:54 2019

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utilities import load_census_data
from fairness_metrics import computeEDFforData
from DNN_model import NeuralNet, training_typical

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

#%% data loading and preprocessing
# load the train dataset
X, y, S = load_census_data('data/adult.data',1)

# Define all the "intersectional groups" to maintain stochastic update of p(y|S) correctly among different batches 
intersectionalGroups = np.unique(S,axis=0) # all intersecting groups, i.e. black-women, white-man etc  

# load the test dataset
test_X, test_y, test_S = load_census_data('data/adult.test',0)

#%%
# data pre-processing
# scale/normalize train & test data and shuffle train data
scaler = StandardScaler().fit(X)
scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
X = X.pipe(scale_df, scaler) 
test_X = test_X.pipe(scale_df, scaler)


X, y, S = sk.utils.shuffle(X, y, S, random_state=0)

X = X.values 
y = y.values 
S = S.values
test_X = test_X.values 
test_y = test_y.values 
test_S = test_S.values

X, dev_X, y, dev_y, S, dev_S = train_test_split(X, y, S, test_size=0.30,stratify=y, random_state=7)

#%%
# deep neural network using pytorch

trainData = torch.from_numpy(X)
trainLabel = torch.from_numpy(y.reshape((-1,1)))

#devData = torch.from_numpy(devData)
                        
testData = torch.from_numpy(test_X)
devData = torch.from_numpy(dev_X)

# hyperparameters
input_size = trainData.size()[1]
hidden1 = 16
hidden2 = 16
hidden3 = 16
output_size = 1
miniBatch = 128 # mini-batch size
num_epochs = 100
learning_rate = 0.001

#%%
import sys
sys.stdout=open("stochastic_typical_clf_out.txt","w")

#%% Training typical model
typical_clf = training_typical(input_size,hidden1,hidden2,hidden3,output_size,learning_rate,num_epochs,trainData,trainLabel,miniBatch) 
    
#%%
# Validate the model
with torch.no_grad():
    devData = Variable(devData.float())
    predictProb = typical_clf(devData)
    predicted = ((predictProb>0.5).numpy()).reshape((-1,))
    Accuracy = sum(predicted == dev_y)/len(dev_y)

# Save results

predictProb = (predictProb.numpy()).reshape((-1,))

print(f"DF classifier dev accuracy: {Accuracy: .3f}")
aucScore = roc_auc_score(dev_y,predictProb)
print(f"DF classifier dev ROC AUC: {aucScore: .3f}")
nn_f1 = f1_score(dev_y,predicted)
print(f"DF classifier dev F1 score: {nn_f1: .2f}")

epsilon_hard,epsilon_soft,gamma_hard,gamma_soft,p_rule_hard,p_rule_soft = computeEDFforData(dev_S,predicted,predictProb,intersectionalGroups)

print(f"DF classifier dev epsilon_hard: {epsilon_hard: .3f}")
print(f"DF classifier dev epsilon_soft: {epsilon_soft: .3f}")
print(f"DF classifier dev gamma_hard: {gamma_hard: .3f}")
print(f"DF classifier dev gamma_soft: {gamma_soft: .3f}")
print(f"DF classifier dev p_rule_hard: {p_rule_hard: .3f}")
print(f"DF classifier dev p_rule_soft: {p_rule_soft: .3f}")
#%%
# Test the model
with torch.no_grad():
    testData = Variable(testData.float())
    predictProb = typical_clf(testData)
    predicted = ((predictProb>0.5).numpy()).reshape((-1,))
    Accuracy = sum(predicted == test_y)/len(test_y)

# Save results

predictProb = (predictProb.numpy()).reshape((-1,))

print(f"DF_Classifier accuracy: {Accuracy: .3f}")
aucScore = roc_auc_score(test_y,predictProb)
print(f"DF_Classifier ROC AUC: {aucScore: .3f}")
nn_f1 = f1_score(test_y,predicted)
print(f"DF_Classifier F1 score: {nn_f1: .2f}")

epsilon_hard,epsilon_soft,gamma_hard,gamma_soft,p_rule_hard,p_rule_soft = computeEDFforData(test_S,predicted,predictProb,intersectionalGroups)

print(f"DF_Classifier epsilon_hard: {epsilon_hard: .3f}")
print(f"DF_Classifier epsilon_soft: {epsilon_soft: .3f}")
print(f"DF_Classifier gamma_hard: {gamma_hard: .3f}")
print(f"DF_Classifier gamma_soft: {gamma_soft: .3f}")
print(f"DF_Classifier p_rule_hard: {p_rule_hard: .3f}")
print(f"DF_Classifier p_rule_soft: {p_rule_soft: .3f}")


   
