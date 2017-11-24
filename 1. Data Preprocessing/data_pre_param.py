#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 00:53:24 2017

@author: Param
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Data.csv')
#print(dataset.head())

x = dataset.iloc[:,:-1].values #Here iloc arg1 - no of rows, arg2 - no of columns
y = dataset.iloc[:,-1:].values #Simple arg2 is 3 instead of -1:


#Handling missing data
from sklearn.preprocessing import Imputer, OneHotEncoder
impute = Imputer(missing_values="NaN", strategy="mean", axis=0) #same as Imputer()
impute = impute.fit(x[:,1:])
x[:,1:] = impute.transform(x[:,1:])

#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder 
le_x = LabelEncoder()
x[:,0] = le_x.fit_transform(x[:,0])
ohe = OneHotEncoder(categorical_features = [0])
x = ohe.fit_transform(x).toarray()

le_y = LabelEncoder()
y = le_x.fit_transform(y)

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_text = sc_x.transform(x_test)
