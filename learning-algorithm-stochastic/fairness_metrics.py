# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:35:23 2020

@author: islam
"""
import pandas as pd
import numpy as np
#%%
# Measure intersectional DF from positive predict probabilities
def differentialFairnessBinaryOutcome(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = 0.0 # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = max(epsilon,abs(np.log(probabilitiesOfPositive[i])-np.log(probabilitiesOfPositive[j]))) # ratio of probabilities of positive outcome
                epsilon = max(epsilon,abs(np.log((1-probabilitiesOfPositive[i]))-np.log((1-probabilitiesOfPositive[j])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = max(epsilonPerGroup) # overall DF of the algorithm 
    return epsilon

#%%
# Measure SP-Subgroup fairness (gamma unfairness) 
def subgroupFairness(probabilitiesOfPositive,alphaSP):
    # input: probabilitiesOfPositive = Pr[D(X)=1|g(x)=1]
    #        alphaG = Pr[g(x)=1]
    # output: gamma-unfairness
    spD = sum(probabilitiesOfPositive*alphaSP) # probabilities of positive class across whole population SP(D) = Pr[D(X)=1]
    gammaPerGroup = np.zeros(len(probabilitiesOfPositive)) # SF per group
    for i in range(len(probabilitiesOfPositive)):
        gammaPerGroup[i] = alphaSP[i]*abs(spD-probabilitiesOfPositive[i])
    gamma = max(gammaPerGroup) # overall SF of the algorithm 
    return gamma

#%%
# mechanisms to make classification outcomes free of disparate impact, that is, 
# to ensure that similar fractions of people from different demographic groups 
# (e.g., males, females) are accepted (or classified as positive) by the classifier.
# More discussion about the disparate impact notion can be found in Sections 1 and 2 of the paper.
# paper: https://arxiv.org/pdf/1507.05259.pdf 
def compute_p_rule(x_control, class_labels,predictProb):

    """ Compute the p-rule based on Doctrine of disparate impact """

    non_prot_all = sum(x_control == 1) # non-protected group
    prot_all = sum(x_control == 0) # protected group
    
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    non_prot_pos = sum(class_labels[x_control == 1] == 1) # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0] == 1) # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    p_rule = min((frac_prot_pos+dirichletAlpha) / (frac_non_prot_pos+concentrationParameter),(frac_non_prot_pos+dirichletAlpha) / (frac_prot_pos+concentrationParameter)) * 100.0
    
    # soft p-rule
    non_prot_pos_soft = sum(predictProb[x_control == 1]) # non_protected in positive class
    prot_pos_soft = sum(predictProb[x_control == 0]) # protected in positive class
    frac_non_prot_pos_soft = float(non_prot_pos_soft) / float(non_prot_all)
    frac_prot_pos_soft = float(prot_pos_soft) / float(prot_all)
    p_rule_soft = min((frac_prot_pos_soft+dirichletAlpha) / (frac_non_prot_pos_soft+concentrationParameter),(frac_non_prot_pos_soft+dirichletAlpha) / (frac_prot_pos_soft+concentrationParameter)) * 100.0
    
    return p_rule,p_rule_soft

#%%
# smoothed empirical differential fairness measurement
def computeEDFforData(protectedAttributes,predictions,predictProb,intersectGroups):
    # compute counts and probabilities
    countsClassOne = np.zeros(len(intersectGroups))
    countsTotal = np.zeros(len(intersectGroups))
    countsClassOne_soft = np.zeros(len(intersectGroups))
    
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    population = len(predictions)
    
    # p-rule specific parameter
    x_control = np.int64(np.ones((len(predictions))))
    
    for i in range(len(predictions)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] = countsTotal[index] + 1
        countsClassOne_soft[index] = countsClassOne_soft[index] + predictProb[i]
        if predictions[i] == 1:
            countsClassOne[index] = countsClassOne[index] + 1
        if protectedAttributes[i,0]==0 and protectedAttributes[i,1]==0 and protectedAttributes[i,2]==0:
            x_control[i] = 0
            
    
    # probability of y given S (p(y=1|S)): probability distribution over merit per value of the protected attributes
    probabilitiesOfPositive_hard = (countsClassOne + dirichletAlpha) /(countsTotal + concentrationParameter)
    probabilitiesOfPositive_soft = (countsClassOne_soft + dirichletAlpha) /(countsTotal + concentrationParameter)
    alphaG_smoothed = (countsTotal + dirichletAlpha) /(population + concentrationParameter)

    epsilon_hard = differentialFairnessBinaryOutcome(probabilitiesOfPositive_hard)
    gamma_hard = subgroupFairness(probabilitiesOfPositive_hard,alphaG_smoothed)
    
    epsilon_soft = differentialFairnessBinaryOutcome(probabilitiesOfPositive_soft)
    gamma_soft = subgroupFairness(probabilitiesOfPositive_soft,alphaG_smoothed)
    
    p_rule,p_rule_soft = compute_p_rule(x_control, predictions,predictProb)
    return epsilon_hard,epsilon_soft,gamma_hard,gamma_soft,p_rule,p_rule_soft