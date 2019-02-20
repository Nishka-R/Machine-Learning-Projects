#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:19:00 2018

@author: nishkaranjan
"""

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# importing the dataset
dataset = pd.read_csv('//Users/nishkaranjan/Desktop/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# Avoiding dummy variable trap: Start of backward elimination
X = X[:,1:]

# splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Building Multiple Linear Regression model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backword Elimination
import statsmodels.formula.api as sm
# We invert the parameters of arr and values so that the column is added at the beginning of the matrix
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#summary presents us with r^2 and p values; we keep removing variables with p value > 0.05
regressor_OLS.summary()
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()