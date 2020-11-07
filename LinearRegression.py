#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:16:38 2020

@author: deep
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Data Preprocessing
training_data_set = pd.read_csv('train.csv')
test_data_set = pd.read_csv('test.csv')

print("Dirty Training Set :", training_data_set.size)
clean_training_set = training_data_set.dropna()
print("Clean Training Set :", clean_training_set.size)

clean_test_set = test_data_set.dropna()

X = np.array(clean_training_set.iloc[:, :-1].values)
#print(X)
Y = np.array(clean_training_set.iloc[:, 1].values)
#print(Y)

X_Test = np.array(clean_test_set.iloc[:, :-1].values)
Y_Test = np.array(clean_test_set.iloc[:, 1].values)

regressor = LinearRegression()
regressor.fit(X,Y)

#Visualizing Training Set
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Size Of House vs Price (Training set)')
plt.xlabel('Size Of House')
plt.ylabel('Price')
plt.show()  

#Visualizing Test Set
plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Test, regressor.predict(X_Test), color = 'blue')
plt.title('Size Of House vs Price (Test set)')
plt.xlabel('Size Of House')
plt.ylabel('Price')
plt.show()  

Y_Pred = regressor.predict(X_Test)

print(regressor.score(X_Test,Y_Test))