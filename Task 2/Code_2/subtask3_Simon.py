# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Description Task 2

#--------------------------------------------------------------------------------------------------
#IMPORTS
#-------
import numpy as np
import pandas as pd
import random
import time as time
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

#--------------------------------------------------------------------------------------------------
#FUNCTIONS
#---------
def print_elapsed_time(starttime):
    time_now=time.perf_counter()
    elapsed_time=time_now-starttime
    print("Time elapsed since start: %.2f s" % elapsed_time)

def prepare_XYmat(features_pd,labels_pd):
    list_pid = features_pd.pid.unique() #get list of all pid's in data
    X=[]
    y=[]
    for pid in list_pid:
        append_X=features_pd.loc[features_pd['pid'] == pid].to_numpy()
        append_X=append_X[2:].flatten()
        append_y=labels_pd.loc[labels_pd['pid']==pid].to_numpy()
        X.append(list(append_X))
        y.append(list(append_y[12:15]))
        X_np=np.array(X)
        y_np=np.array(y)
        #print('Finished pid = %d' % pid)

    return X_np, y_np


#--------------------------------------------------------------------------------------------------
#VARIABLES
#---------
#TODO:IMPORTFILE=['train_features_simpleImpute_mean.csv']     #List of datafiles, computation is performed for each of them.
FILE_FEATURES='train_features_simpleImpute_mean.csv'
FILE_LABELS='Data_2/train_labels.csv'

ALPHAS=[0.01]
KERNELS=['linear']

KFOLD_SPLITS=10
KFOLD_REPEATS=1

#--------------------------------------------------------------------------------------------------
#CODE
#----
data_features=pd.read_csv(FILE_FEATURES)
data_labels=pd.read_csv(FILE_LABELS)

[X_train, Y_train]=prepare_XYmat(data_features,data_labels)

#print(np.shape(X_train))
#print(np.shape(Y_train))

kridge=KernelRidge()
rkf=RepeatedKFold(n_splits=KFOLD_SPLITS,n_repeats=KFOLD_REPEATS,random_state=1234)
paramgrid= {'alpha': ALPHAS, 'kernel': KERNELS }

gscv=GridSearchCV(kridge,param_grid=paramgrid,scoring = 'neg_root_mean_squared_error',n_jobs=2,cv=rkf)
gscv.fit(X_train,y=Y_train)
pd.DataFrame(gscv.cv_results_).to_csv("gscv_results.csv")
Y_predict=gscv.predict(X_train)

[y_row,y_col]=np.shape(Y_predict)
labels_predict_np=np.zeros(y_row,15)
labels_predict_np[12:15]=Y_predict


