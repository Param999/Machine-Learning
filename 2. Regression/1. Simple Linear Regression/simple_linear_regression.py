#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 23:27:33 2017

@author: param
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')
#print(dataset.head(20))
x = dataset.iloc[:,:-1].values #Matrix of features. Independent variable - Years of exp
y = dataset.iloc[:,1].values #Dependent variable - Salary. Vector 
#y = dataset['Salary']
#print(y)

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#Fitting Simple Linear Regression to training set
from sklearn.linear_model import LinearRegression #import LinearRegression class from sklearn
regressor = LinearRegression() #create object of type LinearRegression
regressor.fit(x_train,y_train) #Call fit method 
accuracy = regressor.score(x_test,y_test)
print('Accuracy = %f' %accuracy)
#Predicting the test set results
y_pred = regressor.predict(x_test)

#Visualising the training set results
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train,regressor.predict(x_train))
plt.title('EXP VS Salary (Training Set)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()


#Visualising the test set results
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train,regressor.predict(x_train))
plt.title('EXP VS Salary (Test Set)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()