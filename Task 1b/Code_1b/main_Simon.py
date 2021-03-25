
#--------------------------------------------------------------------------------------------------
# TASK 1b - Intro to Machine Learning ETHZ SS2021
#
# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#
#--------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import RepeatedKFold
from math import sqrt

class data:
    def __init__(self,file,ycol,xcol_start):
        self.dataset=pd.read_csv(file)
        self.np_data=self.dataset.to_numpy()
        #self.X=np.delete(self.np_data,ycol,axis=1)
        self.X=self.np_data[:,xcol_start:]
        self.y=self.np_data[:,ycol]

def prepare_data(linear_X):
    squared_X=np.square(linear_X)
    exp_X=np.exp(linear_X)
    cos_X=np.cos(linear_X)
    const_X=np.ones((linear_X.shape[0],1))
    Phi=np.concatenate((linear_X,squared_X,exp_X,cos_X,const_X),axis=1)
    return Phi

def own_scoring(y, y_pred, **kwargs):
    return sqrt(mean_squared_error(y, y_pred))
#----------------------------------------------------------------------------------------------------------------------
# Read in Data
train_data=data('Data_1b/train.csv',1,2)

ownscore=make_scorer(own_scoring)

Phi=prepare_data(train_data.X)

rkf = RepeatedKFold(n_splits=10, n_repeats=5,random_state=999)

Error=0

report=pd.DataFrame(data=None, columns=["Error"])

for train_index, test_index in rkf.split(Phi):
    X_train = Phi[train_index, :]
    y_train = train_data.y[train_index]
    X_test = Phi[test_index, :]
    y_test = train_data.y[test_index]

    regr=ElasticNet(alpha=0.5,l1_ratio=0.5)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    Error = Error + own_scoring(y_test, y_pred)

meanError=Error/50
print(meanError)
#report = report.append({'alpha_': meanError': meanError}, ignore_index=True)


parameters={'alpha':[0.5],'l1_ratio':[0.5]}

engscv=ElasticNet(max_iter=1000)
gscv=GridSearchCV(engscv,param_grid=parameters,scoring=ownscore,cv=rkf)
gscv.fit(Phi,train_data.y)

gscv_meanError=gscv.error_score
print(gscv_meanError)
gscv_BestError=gscv.best_score_
print(gscv_BestError)
#bestestim_gscv.fit(Phi,train_data.y)
#print(bestestim_gscv.coef_)