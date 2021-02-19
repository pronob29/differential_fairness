# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:03:22 2020

@author: islam
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Loss and optimizer
def fairness_loss(base_fairness,countClass_hat,countTotal_hat):
    # DF-based penalty term
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0) 
    
    theta = (countClass_hat + dirichletAlpha) /(countTotal_hat + concentrationParameter)
    #theta = theta/sum(theta)
    epsilonClass = differentialFairnessBinaryOutcomeTrain(theta)
    return torch.max(zeroTerm, (epsilonClass-base_fairness))
# Loss and optimizer
def sf_loss(base_fairness,countClass_hat,countTotal_hat):
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0) 
    population = sum(countTotal_hat).detach()
    
    theta = (countClass_hat + dirichletAlpha) /(countTotal_hat + concentrationParameter)
    alpha = (countTotal_hat + dirichletAlpha) /(population + concentrationParameter)
    #theta = theta/sum(theta)
    gammaClass = subgroupFairnessTrain(theta,alpha)
    return torch.max(zeroTerm, (gammaClass-base_fairness))

def prule_loss(base_fairness,countClass_hat,countTotal_hat):
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0) 
    
    theta_minority = (countClass_hat[0] + dirichletAlpha) /(countTotal_hat[0] + concentrationParameter)
    theta_majority = (countClass_hat[1] + dirichletAlpha) /(countTotal_hat[1] + concentrationParameter)
    #theta = theta/sum(theta)
    pruleClass = torch.min(theta_minority / theta_majority, theta_majority / theta_minority) * 100.0    
    return torch.max(zeroTerm, (base_fairness-pruleClass))
#%%
# Measure intersectional DF from positive predict probabilities
def differentialFairnessBinaryOutcomeTrain(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = torch.zeros(len(probabilitiesOfPositive),dtype=torch.float)
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0) # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = torch.max(epsilon,torch.abs(torch.log(probabilitiesOfPositive[i])-torch.log(probabilitiesOfPositive[j]))) # ratio of probabilities of positive outcome
                epsilon = torch.max(epsilon,torch.abs((torch.log(1-probabilitiesOfPositive[i]))-(torch.log(1-probabilitiesOfPositive[j])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = torch.max(epsilonPerGroup) # overall DF of the algorithm 
    return epsilon

def subgroupFairnessTrain(probabilitiesOfPositive,alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    spD = sum(probabilitiesOfPositive*alphaSP) # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gammaPerGroup = torch.zeros(len(probabilitiesOfPositive),dtype=torch.float) # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i]*torch.abs(spD-probabilitiesOfPositive[i])
    gamma = torch.max(gammaPerGroup) # overall SF of the algorithm 
    return gamma

#%% empirical count on train set
def computeBatchCounts(protectedAttributes,intersectGroups,predictions):
    # intersectGroups should be pre-defined so that stochastic update of p(y|S) 
    # can be maintained correctly among different batches   
     
    # compute counts for each intersectional group
    countsClassOne = torch.zeros((len(intersectGroups)),dtype=torch.float)
    countsTotal = torch.zeros((len(intersectGroups)),dtype=torch.float)
    for i in range(len(predictions)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] = countsTotal[index] + 1
        countsClassOne[index] = countsClassOne[index] + predictions[i]        
    return countsClassOne, countsTotal

