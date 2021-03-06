# -----------------------------------------------------------------------------
# IML Task_0
# jbaumer
# March 2021
# basic ideas (plus some code fragments) from
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# -----------------------------------------------------------------------------


import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt

## ToDo: find and print RMSE for train data


# get TRAIN data


# ead file train.csv
data_train = np.genfromtxt('task0_sl19d1/train.csv', delimiter=',')

# delete first row (which contains nan values)
data_train = np.delete(data_train, (0), axis=0)

# get number of rows and columns
n_row_train = data_train.shape[0]
n_col_train = data_train.shape[1]

# get id column
data_train_id = data_train[:,0]

# get y column
data_train_Y = data_train[:, 1]

# get x columns
data_train_X = data_train[:, 2:n_col_train]


# get TEST data


# read file test.csv
data_test = np.genfromtxt('task0_sl19d1/test.csv', delimiter=',')

# delete first row (which contains nan values)
# to check where the nan values are: print(np.argwhere(np.isnan(data_test)))
data_test = np.delete(data_test, (0), axis=0)

# get id column
data_test_id = data_test[:,0]

# get number of rows and columns
n_row_test = data_test.shape[0]
n_col_test = data_test.shape[1]

# get x columns
data_test_X = data_test[:, 1:n_col_test]




# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
# this finds a certain f_hat from the function class F
# F: function class holding all linear functions (here)
# f_hat: a*x_1 + b*x_2 + c*x_3 + d*x_4 + ... + j*x_10
# the following line finds the best coeff. (a-j), so that the Y is met as well as possible
regr.fit(data_train_X, data_train_Y)

# Make predictions using the testing set (with found coeff. a-j from training (regr.fit)
data_test_Y_pred = regr.predict(data_test_X)


# Make predictions using the testing set
data_test_Y_pred = regr.predict(data_test_X)

# save stuff in a file
output = np.column_stack((data_test_id, data_test_Y_pred))
np.savetxt('regression_predictions.csv', output, delimiter=',', header="Id,y", comments='')



# Make predictions using the training set
data_train_Y_pred = regr.predict(data_train_X)

#find RSME for train data:
mse_train = mean_squared_error(data_train_Y, data_train_Y_pred)
rmse_train = sqrt(mse_train)
print("Train RMSE:", "\n", rmse_train)