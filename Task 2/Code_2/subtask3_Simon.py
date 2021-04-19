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
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import sys

#--------------------------------------------------------------------------------------------------
#FUNCTIONS
#---------
def print_elapsed_time(starttime):
    time_now=time.perf_counter()
    elapsed_time=time_now-starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)

def prepare_Xmat(features_pd):
    list_pid = features_pd.pid.unique() #get list of all pid's in data
    X=np.empty((0,370))

    i=0
    for pid in list_pid:#[1,10,100,1000,10000,10002,10006,10007,1001,10010]: #
        append_X=features_pd.loc[features_pd['pid'] == pid].to_numpy()
        append_X=append_X[2:].flatten()
        X=np.vstack((X,append_X))

        #i=i+1
        #if(i%1000==0):
        #    print("Finished %d pid's for X matrix..." % i)
    X_np = np.array(X)

    print("Finished X-matrix.")

    return X_np

def prepare_Ymat(labels_pd):
    list_pid = labels_pd.pid.unique() #get list of all pid's in data
    y=np.empty((0,16))

    i=0
    for pid in list_pid:#[1,10,100,1000,10000,10002,10006,10007,1001,10010]: #
        append_y=labels_pd.loc[labels_pd['pid']==pid].to_numpy()
        y=np.vstack((y, append_y))

        #i=i+1
        #if(i%1000==0):
        #    print("Finished %d pid's for y matrix..." % i)
    y_np = y[:,12:16]

    print("Finished y-matrix.")

    return y_np

def compute_Score(TRAIN_FEATURES,TRAIN_LABELS,TEST_FEATURES,KFOLD_SPLITS,KFOLD_REPEATS,starttime,verbose=1):
    if (verbose >= 1):
        print("Execution started.")

    data_features = pd.read_csv(TRAIN_FEATURES)
    data_labels = pd.read_csv(TRAIN_LABELS)
    test_features = pd.read_csv(TEST_FEATURES)

    if (verbose >= 1):
        print("Finished dataimport, starting to prepare XY matrices...")
        print_elapsed_time(starttime)

    X_train = prepare_Xmat(data_features)
    del data_features
    Y_train = prepare_Ymat(data_labels)
    del data_labels
    X_test = prepare_Xmat(test_features)
    del test_features

    if (verbose >= 1):
        print("Finished to prepare X and Y matrices,Shape of X_train and Y_train is: ")
        print(np.shape(X_train))
        print(np.shape(Y_train))
        print("Regression starts...")
        print_elapsed_time(starttime)

    ENreg = ElasticNet(random_state=1234,max_iter=10e4,tol=5e-4)
    rkf = RepeatedKFold(n_splits=KFOLD_SPLITS, n_repeats=KFOLD_REPEATS, random_state=1234)
    paramgrid = {'l1_ratio': L1_RATIO, 'alpha': ALPHAS}
    if (verbose >= 1):
        print("The following parameter Grid is used for regression: ")
        print(paramgrid)

    gscv = GridSearchCV(ENreg, param_grid=paramgrid, scoring='neg_root_mean_squared_error', n_jobs=2, cv=rkf)
    if (verbose >= 1):
        print("  Finished regression-prep, fit and predict starts:")
    gscv.fit(X_train, y=Y_train)
    if (verbose >= 1):
        print("   Fit finished...")
    Y_predict = gscv.predict(X_test)
    if (verbose >= 1):
        print("   predict finished. Preparing result files and outputs...")

    gcsv_results_pd = gscv.cv_results_

    # Creating array of predicted labels
    [y_row, y_col] = np.shape(Y_predict)
    labels_predict_np = np.zeros((y_row, 16))
    labels_predict_np[:, 12:] = Y_predict
    labels_predict_pd = pd.DataFrame(labels_predict_np, columns=data_labels.columns)

    #add additional col to destingiush between different strategies
    gcsv_results_pd['TRAIN_FEATURES']=TRAIN_FEATURES
    gcsv_results_pd['TEST_FEATURES'] = TEST_FEATURES
    gcsv_results_pd['TRAIN_LABELS'] = TRAIN_LABELS

    return labels_predict_pd, gcsv_results_pd


#--------------------------------------------------------------------------------------------------
#VARIABLES
#---------
#TODO:IMPORTFILE=['train_features_simpleImpute_mean.csv']     #List of datafiles, computation is performed for each of them.
FILE_FEATURES='train_features_simpleImpute_mean.csv'
FILE_LABELS='Data_2/train_labels.csv'
TEST_FEATURES='test_features_simpleImpute_constant.csv'

importfiles=[{'name_of_compute': 'mean',
              'FILE_FEATURE':'train_features_simpleImpute_mean.csv',
              'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES':'test_features_simpleImpute_mean.csv'},
             {'name_of_compute': 'median',
             'FILE_FEATURE':'train_features_simpleImpute_median.csv',
             'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES': 'test_features_simpleImpute_median.csv'},
             {'name_of_compute': 'constant',
             'FILE_FEATURE':'train_features_simpleImpute_constant.csv',
             'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES': 'test_features_simpleImpute_constant.csv'}]

#importfiles=[{'name_of_compute': 'mean',
#              'FILE_FEATURE':'train_features_simpleImpute_mean.csv',
#              'FILE_LABELS': 'Data_2/train_labels.csv',
#             'TEST_FEATURES':'test_features_simpleImpute_mean.csv'}]


ALPHAS=[0.01,0.05,0.1,1,10,20,25,50,75,100]
L1_RATIO=[0.1,0.5,0.9,0.95,0.99]
KERNELS=['precomputed']

KFOLD_SPLITS=2
KFOLD_REPEATS=1

verbose=1

current_time_str=datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

#--------------------------------------------------------------------------------------------------
#CODE
#----
time_start_overall=time.perf_counter()


for strat in importfiles:

    print('Starting with some strat...')
    print(strat['name_of_compute'])

    FILE_FEATURES=strat['FILE_FEATURE']
    FILE_LABELS=strat['FILE_LABELS']
    TEST_FEATURES=strat['TEST_FEATURES']

    [labels_predict_pd,this_gscv_results]=compute_Score(FILE_FEATURES,FILE_LABELS,TEST_FEATURES,KFOLD_SPLITS,KFOLD_REPEATS,time_start_overall,1)

    labels_predict_pd.to_csv("predictedLabels_" + current_time_str + "_"+strat['name_of_compute']+ ".csv", index=False)
    pd.DataFrame(this_gscv_results).to_csv("gscv_results" + current_time_str + "_"+strat['name_of_compute']+ ".csv")

    print('Finished.')

