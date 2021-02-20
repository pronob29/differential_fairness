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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score

np.random.seed(7)

#%%
# income predictions on "Census Income" dataset 
# race, gender & nationality selected as protected attributes
## parse the dataset into three dataset: features (X), targets (y) and protected attributes (S)
def load_census_data (path,check):
    column_names = ['age', 'workclass','fnlwgt','education','education_num',
                    'marital_status','occupation','relationship','race','gender',
                    'capital_gain','capital_loss','hours_per_week','nationality','target']
    input_data = (pd.read_csv(path,names=column_names,
                               na_values="?",sep=r'\s*,\s*',engine='python'))
    # sensitive attributes; we identify 'race','gender' and 'nationality' as sensitive attributes
    # note : keeping the protected attributes in the data set, but make sure they are converted to same category as in the S
    input_data['race'] = input_data['race'].map({'Black': 0,'White': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 3})
    input_data['gender'] = (input_data['gender'] == 'Male').astype(int)
    input_data['nationality'] = (input_data['nationality'] == 'United-States').astype(int)
    
    protected_attribs = ['race', 'gender','nationality']
    S = (input_data.loc[:, protected_attribs])
   
    # targets; 1 when someone makes over 50k , otherwise 0
    if(check):
        y = (input_data['target'] == '>50K').astype(int)    # target 1 when income>50K
    else:
        y = (input_data['target'] == '>50K.').astype(int)    # target 1 when income>50K
    
    X = (input_data
         .drop(columns=['target'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape[0]} samples")
    print(f"protected attributes S: {S.shape[0]} samples, {S.shape[1]} attributes")
    return X, y, S

# load the train dataset
X, y, S = load_census_data('data/adult.data',1)

# load the test dataset
test_X, test_y, test_S = load_census_data('data/adult.test',0)

protectedAttributes = test_S.values
originalPredictions = test_y.values


#%%
# train logistic regression as M(X) to replace the original label by the classifier's predictions
def logisticRegressionMx(X,test_X,y,test_y): 
    scaler = StandardScaler().fit(X)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X = X.pipe(scale_df, scaler) 
    test_X = test_X.pipe(scale_df, scaler)
    clfLR = LogisticRegression(C=1.0,random_state=0,solver='liblinear')
    clfLR.fit(X,y) 
    predictions = clfLR.predict(test_X)
# =============================================================================
#     # check the performance of the classifier
#     conf_mat=confusion_matrix(test_y.values, predictions)
#     print(sum(test_y.values==predictions)/len(predictions))
# =============================================================================
    return predictions

predictions = logisticRegressionMx(X,test_X,y,test_y) # mechanism/classiifer's predicted labels on test data

#%%
# smoothed empirical differential fairness measurement
def computeEDFSmoothed(protectedAttributes,predictions):
    S1 = np.unique(protectedAttributes[:,0]) # number of races
    S2 = np.unique(protectedAttributes[:,1]) # number of gender
    S3 = np.unique(protectedAttributes[:,2]) # number of nationalities
    # compute counts and probabilities
    countsClassOne = np.zeros(len(S1)*len(S2)*len(S3)) #each entry corresponds to an intersection, arrays sized by largest number of values 
    countsTotal = np.zeros(len(S1)*len(S2)*len(S3))
    
    numClasses = 2; # for Dirichlet smoothing
    concentrationParameter = 1
    dirichletAlpha = concentrationParameter/numClasses
    
    uniqueA = np.zeros((len(S1)*len(S2)*len(S3),protectedAttributes.shape[1]), dtype=int)
    indx = 0
    for i in S1:
        for j in S2:
            for k in S3:
            #probabilitiesForDF[indx] = probabilitiesClassOne[i,j,k]
             uniqueA[indx,0] = i
             uniqueA[indx,1] = j
             uniqueA[indx,2] = k
             indx += 1
    for i in range(len(predictions)):
        index=np.where((uniqueA==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] = countsTotal[index]+ 1
        countsClassOne[index] = countsClassOne[index] + predictions[i]
        
    #probabilitiesClassOne = countsClassOne/countsTotal
    probabilitiesForDFSmoothed = (countsClassOne + dirichletAlpha) /(countsTotal + concentrationParameter);

    #epsilonSmoothed = differentialFairnessBinaryOutcomeTrain(probabilitiesForDFSmoothed)
    
    population = len(predictions)  
    alphaSP = (countsTotal + dirichletAlpha) /(population + concentrationParameter)
    #gammaSmoothed = subgroupFairness(probabilitiesForDFSmoothed,alphaSP)
    
    return probabilitiesForDFSmoothed,alphaSP

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
    return epsilon,epsilonPerGroup
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
    return gamma,gammaPerGroup

#%%
# Measurement of inequality: Gini index or Gini coefficient
def computeGiniIndex(alphaSP,unfairnessPerGroup):
    gini_tmp = 0.0 
    mu = 0.0
    for i in  range(len(alphaSP)):
        mu += alphaSP[i]*unfairnessPerGroup[i]
        for j in range(len(alphaSP)):
            gini_tmp += alphaSP[i]*alphaSP[j]*abs(unfairnessPerGroup[i]-unfairnessPerGroup[j])
    G = gini_tmp/(2*mu) # Gini index
    return G

import sys
sys.stdout=open("Adult_Gini_index.txt","w")
#%% compute for data 
probabilitiesForDFSmoothed,alphaSP=computeEDFSmoothed(protectedAttributes,originalPredictions)
epsilon,epsilonPerGroup=differentialFairnessBinaryOutcome(probabilitiesForDFSmoothed)
gamma,gammaPerGroup=subgroupFairness(probabilitiesForDFSmoothed,alphaSP)
# compute Gini index for DF:
G_DF = computeGiniIndex(alphaSP,epsilonPerGroup)
print(f"data Gini-DF: {G_DF: .3f}")
# compute Gini index for SF:
G_SF = computeGiniIndex(alphaSP,gammaPerGroup)
print(f"data Gini-SF: {G_SF: .3f}")

#%% compute for algorithm
probabilitiesForDFSmoothed,alphaSP=computeEDFSmoothed(protectedAttributes,predictions)
epsilon,epsilonPerGroup=differentialFairnessBinaryOutcome(probabilitiesForDFSmoothed)
gamma,gammaPerGroup=subgroupFairness(probabilitiesForDFSmoothed,alphaSP)
# compute Gini index for DF:
G_DF = computeGiniIndex(alphaSP,epsilonPerGroup)
print(f"algorithm Gini-DF: {G_DF: .3f}")
# compute Gini index for SF:
G_SF = computeGiniIndex(alphaSP,gammaPerGroup)
print(f"algorithm Gini-SF: {G_SF: .3f}")
