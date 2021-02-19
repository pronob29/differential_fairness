# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:30:17 2020

@author: islam
"""
import pandas as pd
import numpy as np

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
    return X, y, S

