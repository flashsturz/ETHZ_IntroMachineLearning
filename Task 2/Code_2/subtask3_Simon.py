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
from score_submission import get_score
from prepare_matrices import prepare_Xmat, prepare_Ymat
import sys

#--------------------------------------------------------------------------------------------------
#FUNCTIONS
#---------
def print_elapsed_time(starttime):
    time_now=time.perf_counter()
    elapsed_time=time_now-starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)

def compute_Estimator(X_train,Y_train,KFOLD_SPLITS,KFOLD_REPEATS,starttime,verbose=1):

    if (verbose >= 1):
        print("Regression starts...")
        print_elapsed_time(starttime)

    ENreg = ElasticNet(random_state=1234,max_iter=10e4,tol=5e-4)
    rkf = RepeatedKFold(n_splits=KFOLD_SPLITS, n_repeats=KFOLD_REPEATS, random_state=1234)
    paramgrid = {'l1_ratio': L1_RATIO, 'alpha': ALPHAS}
    if (verbose >= 1):
        print("The following parameter Grid is used for regression: ")
        print(paramgrid)

    gscv = GridSearchCV(ENreg, param_grid=paramgrid, scoring='neg_root_mean_squared_error', n_jobs=4, cv=rkf)
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
FILE_FEATURES='train_features_simpleImpute_mean.csv'
FILE_LABELS='Data_2/train_labels.csv'
TEST_FEATURES='test_features_simpleImpute_constant.csv'

importfiles=[{'name_of_compute': 'mean',
              'Xmat_file_given': False,
              'FILE_FEATURE':'train_features_simpleImpute_mean.csv',
              'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES':'test_features_simpleImpute_mean.csv'},
             {'name_of_compute': 'median',
              'Xmat_file_given': False,
             'FILE_FEATURE':'train_features_simpleImpute_median.csv',
             'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES': 'test_features_simpleImpute_median.csv'},
             {'name_of_compute': 'constant',
              'Xmat_file_given': False,
             'FILE_FEATURE':'train_features_simpleImpute_constant.csv',
             'FILE_LABELS': 'Data_2/train_labels.csv',
             'TEST_FEATURES': 'test_features_simpleImpute_constant.csv'}]



ALPHAS=[0.01,0.1,1,10,25,50,75,100]
L1_RATIO=[0.01,0.1,0.9,0.95,0.99]

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
        data_features = pd.read_csv(TRAIN_FEATURES)
        test_features = pd.read_csv(TEST_FEATURES)

        if (verbose >= 1):
            print("Starting to prepare X matrices...")
            print_elapsed_time(starttime)

        X_train = prepare_Xmat(data_features)
        del data_features
        X_test = prepare_Xmat(test_features)
        del test_features

    data_labels = pd.read_csv(TRAIN_LABELS)
    Y_train = prepare_Ymat(data_labels)
    del data_labels

    if (verbose >= 1):
        print("Finished to prepare X and Y matrices,Shape of X_train and Y_train is: ")
        print(np.shape(X_train))
        print(np.shape(Y_train))

    [this_gscv_results,best_estim]=compute_Estimator(strat['name_of_compute'],X_train,Y_train,KFOLD_SPLITS,KFOLD_REPEATS,time_start_overall,1)

    #Predicting the test-labels and printing them to a csv-file:
    #   Using best_estim given by compute_Estimator()
    #   and provided Test_features in X_test.

    Y_predict = best_estim.predict(X_test)

    if (verbose >= 1):
        print("   predict finished. Preparing result files and outputs...")
        print_elapsed_time(starttime)

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
    else:
        gscv_results=gscv_results.append(this_gscv_results)

    print('Finished this Strat.')

gscv_results['overall_ranking'] = gscv_results['mean_test_score'].rank(method='min')
pd.DataFrame(gscv_results).to_csv("gscv_results_" + current_time_str +".csv")

print('Finished full execution.')
print_elapsed_time(time_start_overall)