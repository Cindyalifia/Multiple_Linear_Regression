# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 00:29:14 2019

@author: PERSONALISE NOTEBOOK
"""

#Multiple Linear Regression 

#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[: , 4].values

#Encoding categorical data , change text to number
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() 
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) 
#encode the dummy variables for column coutry
onehotencoder = OneHotEncoder(categorical_features = [3]) 
X = onehotencoder.fit_transform(X).toarray() 

#Splitting the dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scalling , library has take it 

#Fitting Multiple Linear Regresion to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train) # fit regressor to the Trining Set

#Predicting the Test set results
y_pred = regressor.predict(X_test)


