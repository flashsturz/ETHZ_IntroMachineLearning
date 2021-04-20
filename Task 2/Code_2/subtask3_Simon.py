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
from sklearn.metrics import r2_score, make_scorer

from datetime import datetime
import sys

#--------------------------------------------------------------------------------------------------
#FUNCTIONS
#---------
def print_elapsed_time(starttime):
    time_now=time.perf_counter()
    elapsed_time=time_now-starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)

def task3_score(y, y_pred, **kwargs):
    return np.mean(0.5 + 0.5 * np.maximum(0, r2_score(y, y_pred)))

task3_scorer=make_scorer(task3_score,)

def prepare_Xmat(features_pd):
    list_pid = features_pd.pid.unique() #get list of all pid's in data
    X=np.empty((0,420))

    i=0
    for pid in list_pid:#[1,10,100,1000,10000,10002,10006,10007,1001,10010]: #
        append_X=features_pd.loc[features_pd['pid'] == pid].to_numpy()
        append_X=append_X[:,2:].flatten()
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

def compute_Estimator(X_train,Y_train,KFOLD_SPLITS,KFOLD_REPEATS,starttime,verbose=1):

    if (verbose >= 1):
        print("Regression starts...")
        print_elapsed_time(starttime)

    ENreg = ElasticNet(random_state=1234,max_iter=10e5,tol=1e-4)
    paramgrid = {'l1_ratio': L1_RATIO, 'alpha': ALPHAS}

    rkf = RepeatedKFold(n_splits=KFOLD_SPLITS, n_repeats=KFOLD_REPEATS, random_state=1234)

    if (verbose >= 1):
        print("The following parameter Grid is used for regression: ")
        print(paramgrid)

    gscv = GridSearchCV(ENreg, param_grid=paramgrid, scoring=task3_scorer, n_jobs=4, cv=rkf)
    if (verbose >= 1):
        print("  Finished regression-prep, fit and predict starts:")
        print_elapsed_time(starttime)
    gscv.fit(X_train, y=Y_train)
    if (verbose >= 1):
        print("   Fit finished...")
        print_elapsed_time(starttime)

    gcsv_results_pd = pd.DataFrame(gscv.cv_results_)

    return gcsv_results_pd, gscv.best_estimator_


#--------------------------------------------------------------------------------------------------
#VARIABLES
#---------

big_list=[{'name_of_compute': 'mean',
              'Xmat_file_given': False,
              'FILE_FEATURE':'ImputedFiles/train_features_simpleImpute_mean.csv',
              'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES':'ImputedFiles/test_features_simpleImpute_mean.csv'},
             {'name_of_compute': 'constant',
              'Xmat_file_given': False,
             'FILE_FEATURE':'ImputedFiles/train_features_simpleImpute_constant.csv',
             'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES': 'ImputedFiles/test_features_simpleImpute_constant.csv'}]

acer_list=[{'name_of_compute': 'mean',
              'Xmat_file_given': False,
              'FILE_FEATURE':'ImputedFiles/train_features_simpleImpute_mean.csv',
              'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES':'ImputedFiles/test_features_simpleImpute_mean.csv'},
             {'name_of_compute': 'constant',
              'Xmat_file_given': False,
             'FILE_FEATURE':'ImputedFiles/train_features_simpleImpute_constant.csv',
             'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES': 'ImputedFiles/test_features_simpleImpute_constant.csv'}] #TODO:Add other imp methods.

importfiles=acer_list


ALPHAS=[0.75,1]
L1_RATIO=[0.09,0.1,0.2]

KFOLD_SPLITS=2
KFOLD_REPEATS=1

verbose=1

current_time_str=datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

#--------------------------------------------------------------------------------------------------
#CODE
#----
time_start_overall=time.perf_counter()

Is_first_execution=True
for strat in importfiles:

    print('Starting with some strat...')
    print(strat['name_of_compute'])

    FILE_FEATURES = strat['FILE_FEATURE']
    FILE_LABELS = strat['FILE_LABELS']
    TEST_FEATURES = strat['TEST_FEATURES']

    if strat['Xmat_file_given']:
        X_train=pd.read_csv(FILE_FEATURES).to_numpy()
        X_test =pd.read_csv(TEST_FEATURES).to_numpy()
    else:
        data_features = pd.read_csv(FILE_FEATURES)
        test_features = pd.read_csv(TEST_FEATURES)

        if (verbose >= 1):
            print("Starting to prepare X matrices...")
            print_elapsed_time(time_start_overall)

        X_train = prepare_Xmat(data_features)
        del data_features
        X_test = prepare_Xmat(test_features)
        del test_features

    data_labels = pd.read_csv(FILE_LABELS)
    Y_train = prepare_Ymat(data_labels)
    data_labels

    if (verbose >= 1):
        print("Finished to prepare X and Y matrices,Shape of X_train and Y_train is: ")
        print(np.shape(X_train))
        print(np.shape(Y_train))

    [this_gscv_results,best_estim]=compute_Estimator(X_train,Y_train,KFOLD_SPLITS,KFOLD_REPEATS,time_start_overall,1)

    #Predicting the test-labels and printing them to a csv-file:
    #   Using best_estim given by compute_Estimator()
    #   and provided Test_features in X_test.

    Y_predict = best_estim.predict(X_test)

    if (verbose >= 1):
        print("   predict finished. Preparing result files and outputs...")
        print_elapsed_time(time_start_overall)

    # Creating array of predicted labels
    [y_row, y_col] = np.shape(Y_predict)
    labels_predict_np = np.zeros((y_row, 16))
    labels_predict_np[:, 12:] = Y_predict
    #Print to file:
    labels_predict_pd = pd.DataFrame(labels_predict_np, columns=data_labels.columns)
    labels_predict_pd.to_csv("predictedLabels_" + current_time_str + "_"+strat['name_of_compute']+ ".csv", index=False)

    #append GridSearch_results:
    this_gscv_results['Name'] = strat['name_of_compute']
    if Is_first_execution:
        gscv_results=this_gscv_results
        Is_first_execution=False
    else:
        gscv_results=gscv_results.append(this_gscv_results)

    print('Finished this Strat.')

gscv_results['overall_ranking'] = gscv_results['mean_test_score'].rank(method='min')
pd.DataFrame(gscv_results).to_csv("gscv_results_" + current_time_str +".csv")

print('Finished full execution.')
print_elapsed_time(time_start_overall)