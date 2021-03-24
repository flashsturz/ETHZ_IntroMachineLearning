
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

rkf = RepeatedKFold(n_splits=10, n_repeats=10,random_state=999)
regr=ElasticNet(random_state=1234)
parameters={'l1_ratio':[0.1,0.3,0.5,0.7,0.9,0.95,0.98,0.99],'alpha':[0.001,0.01,0.1,0.25,0.5,1,10,100]}

gscv = GridSearchCV(regr, param_grid=parameters,cv=rkf,scoring=ownscore,verbose=-1)
gscv.fit(Phi,train_data.y)

pd.DataFrame(gscv.cv_results_).to_csv("output.csv")
best_estim=gscv.best_estimator_
print("Best Estimator Parameters found by GridSearch:")
print(gscv.best_params_)

print(best_estim.coef_)
pd.DataFrame(gscv.best_estimator_.coef_).to_csv("Estimtor_coef.csv")

regrbest=ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9,0.95,0.98,0.99],random_state=4321)
regrbest.fit(Phi,train_data.y)

print(regrbest.coef_)