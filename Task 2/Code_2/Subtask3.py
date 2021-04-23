# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
# -----------------
# Description Task 3

# --------------------------------------------------------------------------------------------------
# IMPORTS
# -------
import numpy as np
import pandas as pd
import time as time
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer

from datetime import datetime


# --------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------

def print_elapsed_time(starttime):
    time_now = time.perf_counter()
    elapsed_time = time_now - starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)


def task3_score(y, y_pred, **kwargs):
    return np.mean(0.5 + 0.5 * np.maximum(0, r2_score(y, y_pred)))

task3_scorer = make_scorer(task3_score)

def prepare_Xmat(features_pd):
    list_pid = features_pd.pid.unique()  # get list of all pid's in data
    X = np.empty((0, 420))

    i = 0
    for pid in list_pid:  # [1,10,100,1000,10000,10002,10006,10007,1001,10010]: #
        append_X = features_pd.loc[features_pd['pid'] == pid].to_numpy()
        append_X = append_X[:, 2:].flatten()
        X = np.vstack((X, append_X))

        # i=i+1
        # if(i%1000==0):
        #    print("Finished %d pid's for X matrix..." % i)
    X_np = np.array(X)

    print("Finished X-matrix.")

    return X_np


def prepare_Ymat(labels_pd):
    list_pid = labels_pd.pid.unique()  # get list of all pid's in data
    y = np.empty((0, 16))

    i = 0
    for pid in list_pid:  # [1,10,100,1000,10000,10002,10006,10007,1001,10010]: #
        append_y = labels_pd.loc[labels_pd['pid'] == pid].to_numpy()
        y = np.vstack((y, append_y))

        # i=i+1
        # if(i%1000==0):
        #    print("Finished %d pid's for y matrix..." % i)
    y_np = y[:, 12:16]

    print("Finished y-matrix.")

    return y_np


def compute_Estimator(X_train, Y_train, KFOLD_SPLITS, KFOLD_REPEATS, starttime, verbose=1):
    if verbose >= 1:
        print("Regression starts...")
        # print_elapsed_time(starttime)

    ALPHAS = [1.1] # 0.95
    L1_RATIO = [0.25] # 0.15

    ENreg = ElasticNet(random_state=1234, max_iter=10e5, tol=1e-4)
    paramgrid = {'l1_ratio': L1_RATIO, 'alpha': ALPHAS}

    rkf = RepeatedKFold(n_splits=KFOLD_SPLITS, n_repeats=KFOLD_REPEATS, random_state=1234)

    if verbose >= 1:
        print("The following parameter Grid is used for regression: ")
        print(paramgrid)

    gscv = GridSearchCV(ENreg, param_grid=paramgrid, scoring=task3_scorer, n_jobs=-1, cv=rkf)
    if verbose >= 1:
        print("  Finished regression-prep, fit and predict starts:")
        # print_elapsed_time(starttime)
    print("type of Y_train: ", type(Y_train))
    print("Shape of Y_train", np.shape(Y_train))

    gscv.fit(X_train, Y_train)
    if verbose >= 1:
        print("   Fit finished...")
        # print_elapsed_time(starttime)

    gcsv_results_pd = pd.DataFrame(gscv.cv_results_)

    print("Grid Search Best Score: \n", gscv.best_score_)
    print("Grid Search Best Estimator: \n", gscv.best_estimator_)

    return gcsv_results_pd, gscv.best_estimator_

def solveSubtask3(train_features_pd,test_features_pd,train_labels_pd,start_time,verbose=1):

    if verbose >= 1:
        print("Subtask3 starts...")
        print("Starting to prepare things...")

    X_train = prepare_Xmat(train_features_pd)
    X_test = prepare_Xmat(test_features_pd)
    Y_train = prepare_Ymat(train_labels_pd)

    if verbose >= 1:
        print("Preparation finished. Regression starts... ")

    KFOLD_SPLITS = 2
    KFOLD_REPEATS = 1

    [this_gscv_results, best_estim] = compute_Estimator(X_train,
                                                        Y_train,
                                                        KFOLD_SPLITS,
                                                        KFOLD_REPEATS,
                                                        start_time,
                                                        1)

    Y_predict = best_estim.predict(X_test)

    # Creating array of predicted labels
    [y_row, y_col] = np.shape(Y_predict)
    labels_predict_np = np.zeros((y_row, 16))
    labels_predict_np[:, 12:] = Y_predict

    # Output
    labels_predict_pd = pd.DataFrame(labels_predict_np, columns=train_labels_pd.columns)

    return labels_predict_pd