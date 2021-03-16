
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
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge
import time as time
import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

from math import sqrt

class data:
    def __init__(self,file,ycol,xcol_start):
        self.dataset=pd.read_csv(file)
        self.np_data=self.dataset.to_numpy()
        #self.X=np.delete(self.np_data,ycol,axis=1)
        self.X=self.np_data[:,xcol_start:]
        self.y=self.np_data[:,ycol]

def own_scoring(y, y_pred, **kwargs):
    return sqrt(mean_squared_error(y, y_pred))
#----------------------------------------------------------------------------------------------------------------------
# Read in Data
train_data=data('Data_1a/train.csv',0,1)

ownscore=make_scorer(own_scoring)

score=np.array([])
report=pd.DataFrame(data=None, columns=["Lambda","Solver","Tolerance","Score","ElapsedTime [s]"])

totaltime_start=time.perf_counter()

CV_lambda=[0.1,1,10,100,200]
for _lambda in CV_lambda:
    time_start=time.perf_counter()
    linmod=Ridge(alpha=_lambda, max_iter=5000)

    #parameters = {'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga'], 'tol': [1e-3]}
    parameters = {'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga'], 'tol': [1e-5,5e-5,1e-4,3e-4,7e-4,1e-3,3e-3,5e-3,7e-3,9e-3,1e-2,3e-2,5e-2,7e-2] }

    rkf = RepeatedKFold(n_splits=10, n_repeats=100,random_state=999)
    gscv = GridSearchCV(linmod, param_grid=parameters,cv=rkf,scoring=ownscore)
    gscv.fit(train_data.X,train_data.y)

    this_results=pd.Series(gscv.cv_results_)
    time_end = time.perf_counter()
    this_bestscore=this_results['mean_test_score'][np.argmin(this_results['mean_test_score'])]
    ind_minscore=np.argmin(this_results['mean_test_score'])

    solver=this_results['params'][ind_minscore]
    score=np.append(score,this_bestscore)
    thisreport = {'Lambda': _lambda, 'Solver': solver['solver'], 'Tolerance': solver['tol'], 'Score': this_bestscore, 'ElapsedTime [s]': time_end-time_start}

    report=report.append(thisreport, ignore_index=True)

    #print('Best Score for Lambda= %.1f was = %.9f Using %s.' % (_lambda,this_bestscore,solver) )

pd.DataFrame(score).to_csv("output.csv",header=None,index=None)

totaltime_end=time.perf_counter()
print(report)
totaltime=totaltime_end-totaltime_start
pd.DataFrame(report).to_csv("report.csv")
print('Best scores were saved in ouput.csv. Total time: %s' % str(datetime.timedelta(seconds=totaltime)) )