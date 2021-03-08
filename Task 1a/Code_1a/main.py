
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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

file = 'train.csv' 
# Read in Data
data = np.genfromtxt(../../Data_1a/file, dtype=float, delimiter=',', skip_header = 1)