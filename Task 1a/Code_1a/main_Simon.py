
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
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from math import sqrt

class data:
    def __init__(self,file,ycol,xcol_start):
        self.dataset=pd.read_csv(file)
        self.np_data=self.dataset.to_numpy()
        #self.X=np.delete(self.np_data,ycol,axis=1)
        self.X=self.np_data[:,xcol_start:]
        self.y=self.np_data[:,ycol]

#----------------------------------------------------------------------------------------------------------------------
# Read in Data
train_data=data('Data_1a/train.csv',0,1)

score=np.array([])

CV_lambda=[0.1,1,10,100,200]
for _lambda in CV_lambda:
    linmod=Ridge(alpha=_lambda,tol=1e-3)
    linmod.fit(train_data.X,train_data.y)

    kf = KFold(n_splits=10, shuffle=False)

    Error=0

    for train_index, test_index in kf.split(train_data.X):
        X_train=train_data.X[train_index,:]
        y_train=train_data.y[train_index]
        X_test = train_data.X[test_index,:]
        y_test = train_data.y[test_index]

        linmod = Ridge(alpha=_lambda, tol=1e-4)
        linmod.fit(X_train, y_train)

        y_pred=linmod.predict(X_test)

        Error=Error+sqrt(mean_squared_error(y_test,y_pred))

    score=np.append(score,Error/kf.get_n_splits(train_data.X))


#print("Score is: ")
#print(score)

pd.DataFrame(score).to_csv("output.csv",header=None,index=None)



