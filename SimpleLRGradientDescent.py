#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:33:13 2020

@author: deep
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(alpha, X, Y, ep = 0.0001, max_iter = 5000):
  
  print('Gradient Descent Model Begins : ')
  
  converged = False
  iter = 0
  #m = X.shape
  #print(m)
  
  t0 = 0 #np.random.random(X.shape[1])
  t1 = 1 #np.random.random(X.shape[1])
  print('t0 : ', t0 ,', t1 :', t1)
  
  #Total Error
  J = sum([( t0  + t1*X[i] - Y[i] )**2 for i in range (X.size) ])
  print('Initial Error J(theta) : ', J)
  
  while not converged: 
    #computing gradient (d/d_theta J(theta)) for each training sample
    grad0 = 1.0/X.size * sum([( t0  + t1*X[i] - Y[i] ) for i in range (X.size)])
    grad1 = 1.0/X.size * sum([( t0  + t1*X[i] - Y[i] )*X[i] for i in range (X.size)])
    
    #Updating theta_temp
    temp0 = t0 - alpha*grad0
    temp1 = t1 - alpha*grad1
    
    #updating theta
    t0 = temp0
    t1 = temp1
    
    #mean_squared_error
    E = sum([( t0  + t1*X[i] - Y[i] )**2 for i in range (X.size) ])
    #print('Mean_Squared Error : ', E)
    
    if (J-E) <= ep:
      print('Final Error : ', E)
      print('Converged at Iteration : ', iter)
      converged = True
    
    J = E
    #print(J)
    iter = iter + 1
    
    if(iter == max_iter):
      print('Maximum Iterations Reached')
      converged = True
      
  return t0,t1
  


alpha = float(input("Enter the value for alpha (Learning Rate) : "))

#Reading the test and training dataset
training_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

#Cleaning the Datasets
clean_training_set = training_dataset.dropna()
clean_test_set = test_dataset.dropna()

X = np.array(clean_training_set.iloc[:, :-1].values)
#print(X)
Y = np.array(clean_training_set.iloc[:, 1].values)
#print(Y)

X_test = np.array(clean_test_set.iloc[:, :-1].values)
Y_test = np.array(clean_test_set.iloc[:, 1].values)

#calling the gradient_descent function and getting the intercepts theta0 and theta1
theta0, theta1 = gradientDescent(alpha, X, Y)

print('Final theta0 :', theta0 ,', theta1 ; ', theta1)

for i in range (X.size):
  Y_train_predict = theta0 + theta1*X

#Visualizing Training Set
plt.scatter(X, Y, color = 'red')
plt.plot(X, Y_train_predict, color = 'blue')
plt.title('Size Of House vs Price (Train set)')
plt.xlabel('Size Of House')
plt.ylabel('Price')
plt.show()


for i in range (X_test.size):
  Y_test_predict = theta0 + theta1*X_test
  

#Visualizing Test Set
plt.scatter(X, Y, color = 'red')
plt.plot(X_test, Y_test_predict, color = 'blue')
plt.title('Size Of House vs Price (Test set)')
plt.xlabel('Size Of House')
plt.ylabel('Price')
plt.show()




