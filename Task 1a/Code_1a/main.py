#--------------------------------------------------------------------------------------------------
# TASK 1a - Intro to Machine Learning ETHZ SS2021
#
# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Goal of this task is to perform 10-fold cross-validation (CV) with ridge regression for a given set of lambda-values
# and report the root mean squared error (RMSE) averaged over the 10 test-folds.
# The linear regression should be performed on the original features (e.g. no data transformation etc.)
#--------------------------------------------------------------------------------------------------

import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

file = 'train.csv' 
random.seed(999)

# Declare Arrays used
rmse_lambdas = [] # Output Vector

# Read in Data
data = np.genfromtxt('../Data_1a/' + file, dtype=float, delimiter=',', skip_header = 1)

x = data[0:, 1:]
y = data[0:, 0]

lambdas = [0.1, 1, 10, 100, 200] # Ridge regression parameters

# Split Data Set
kf = KFold(n_splits = 10)

for _lambda in lambdas:
    rmse = [] # Root Mean Squared Error for single CV Iteration, reset to zero each time before running the 10-fold cross validation
    for train, test in kf.split(x): # Creates the Cross Validation Folds
        x_train = x[train]
        y_train = y[train]
        
        x_val = x[test]
        y_val = y[test]
        
        clf = Ridge(alpha = _lambda) # Initialize Ridge Regression
        clf.fit(x_train, y_train) # Fit Training Data of Cross Validation
        y_pred = clf.predict(x_val) # Predict output on test data of Cross Validation
        rmse.append(np.sqrt(mean_squared_error(y_val, y_pred))) # Calculate the RMSE for this iteration and append to list
        
    avg_rmse = np.mean(rmse) # after iterating through all 10 different folds, average over the sum of the rmse_values
    rmse_lambdas.append(avg_rmse) # create vector containing the average RMSE for the different lambda-values
    
np.savetxt('avg_rmse_lambdas.csv', rmse_lambdas, fmt=['%.19f'], delimiter=',', comments='',)