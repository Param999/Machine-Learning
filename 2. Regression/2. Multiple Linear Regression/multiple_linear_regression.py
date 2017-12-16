#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:22:06 2017

@author: param
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression 
import statsmodels.formula.api as sm

#Import dataset
dataset = pd.read_csv('50_Startups.csv')
#print(dataset.head())

x = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,4].values


#Encoding catagorical data
le_x = LabelEncoder()
x[:,-1] = le_x.fit_transform(x[:,-1])
ohe = OneHotEncoder(categorical_features = [3])
x = ohe.fit_transform(x).toarray()

#Avoiding dummy variable trap
x = x[:,1:]

#Splitting the dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Fitting Simple Linear Regression to training set
regressor = LinearRegression() #create object of type LinearRegression
regressor.fit(x_train,y_train) #Call fit method 
accuracy = regressor.score(x_test,y_test)
print('Accuracy = %f' %accuracy)
#Predicting the test set results
y_pred = regressor.predict(x_test)

#Building optimal model using Backward elimination
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis =1)

x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()  
regressor_OLS.summary()

x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()  
regressor_OLS.summary()

x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()  
regressor_OLS.summary()

x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog= y,exog= x_opt).fit()  
regressor_OLS.summary()