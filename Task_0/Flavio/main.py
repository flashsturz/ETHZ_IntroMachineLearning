# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# IML Introduction to Machine Learning FS 21 ETH Zurich
# Task 0: Regression
# Flavio Regenass, March 2021
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

file = 'test.csv' # file can be either train.csv or test.csv

# Declare Arrays used
x = []
y = []
y_pred = []
Id = []

# Read in Data
data = np.genfromtxt(file, dtype=float, delimiter=',', skip_header = 1)
    
if file == 'train.csv':
    for row in data:
        Id.append(int(row[0]))
        y.append(row[1])
        x.append(list(row[2:]))
    reg = LinearRegression().fit(x, y) # Perform Linear Regression (Least Squares)
    y_pred = reg.predict(x)
    RMSE = mean_squared_error(y, y_pred)**0.5
    print(RMSE)
        
elif file ==  'test.csv':
    for row in data:
        Id.append(int(row[0]))
        x.append(list(row[1:]))
    y_pred = reg.predict(x) # Output prediction
    output = np.column_stack((Id, y_pred)) # Horizontally stack the two columns ID and y_pred
    np.savetxt('Regression_Predictions.csv', output, fmt=['%.1d','%.9f'], delimiter=',', header = "Id,y", comments='',)
    

        



