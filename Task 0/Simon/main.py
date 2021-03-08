#--------------------------------------------------------------------------------------------------
# TASK 0 - Intro to Machine Learning ETHZ SS2021
#
# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# The goal of this task is to predict a y-vector from a given set of x values.
# The predictor is learned using a training set given.
#--------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#reading in the training data and preparing the model
train_set=pd.read_csv('data/train.csv')
train_set['x0']=1

np_train=train_set.to_numpy()
np_train_X=np.delete(np_train,[0,1],axis=1)
np_train_y=np_train[:,0]
reg = LinearRegression().fit(np_train_X, np_train_y)

#Evaluation of the model using RMSE.
y_pred_train = reg.predict(np_train_X)
score_train = mean_squared_error(np_train_y, y_pred_train, squared=False)

#print("The score (calculated with RMSE)is: %.2f" % score_train)

#Reading the test data and predicting the output y_pred
test_set=pd.read_csv('data/test.csv')
test_set['x0']=1

np_test=test_set.to_numpy()
np_test_X=np.delete(np_test,0,axis=1)

y_pred=reg.predict(np_test_X)
id=test_set['Id'].to_numpy()

#Writing the output to a csv file.
pred_output=pd.DataFrame({'Id':id, 'y':y_pred})
#print(pred_output)                     #debugging
pred_output.to_csv('regression_predictions.csv',index=False)







