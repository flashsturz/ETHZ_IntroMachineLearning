# -----------------------------------------------------------------------------
# IML Task_0
# jbaumer
# March 2021
# basic ideas (plus some code fragments) from
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html
# -----------------------------------------------------------------------------
# task: Your task is to perform 10-fold cross-validation with
# ridge regression for each value of λ given above and report
# the Root Mean Squared Error (RMSE) averaged over the 10 test folds.
#
# In other words, for each λ, you should train a
# ridge regression 10 times leaving out a different fold each time,
# and report the average of the RMSEs on the left-out folds.

# create repeated k-fold object (n_folds = 10, repeated over whole dataset (150), random_state - any int
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# get TRAIN data
filepath_train = '../Data_1a/train.csv'

# read file train.csv, skip first row (which contains nan values)
data_train = np.genfromtxt(filepath_train, delimiter=',', skip_header=1)
# get number of rows and columns
n_row_train = data_train.shape[0]
n_col_train = data_train.shape[1]
# get y column
data_train_Y = data_train[:, 0]
# get x columns
data_train_X = data_train[:, 1:n_col_train]

# define all lamdas (acc. to exc)
lambda_full_set = [0.1, 1, 10, 100, 200]

rkf_object = RepeatedKFold(n_splits=10, n_repeats=150, random_state=42)

rmse_avg_lambdas = []
# Creates the Cross Validation Folds
for lmbd in lambda_full_set:

    for train_index, test_index in rkf_object.split(data_train_X): # train index and test index are complementary

        #init vector with all rmse values
        rmse_vec = []

        # get train data
        x_train = data_train_X[train_index]
        y_train = data_train_Y[train_index]

        # get test data
        x_test = data_train_X[test_index]
        y_test = data_train_Y[test_index] # 'real' y_value (for this specific test index)


        # create ridge regression object
        clf = Ridge(alpha=lmbd) #here: loop over all lambdas
        # fit data to model
        clf.fit(x_train, y_train)
        # find predicted y
        y_pred = clf.predict(x_test)

        # find root mean squared error (RMSE) betw. y_test and y_pred
        RMSE_per_fold = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_vec.append(RMSE_per_fold)
    # find avg of all RMSE_per_fold
    rmse_mean = np.mean(RMSE_per_fold)

    rmse_avg_lambdas.append(rmse_mean)

# save rmse_avg_lambdas in file
np.savetxt('rmse_avg_lambdas_jbaumer.csv', rmse_avg_lambdas, fmt=['%.19f'], delimiter=',', comments='',)