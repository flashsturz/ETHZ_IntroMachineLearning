# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Goal of this task is to first impute the missing values in the data set and then perform two different binary classification tasks and one regression task.
# This file solves Subtask 1
#--------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import time

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import roc_auc_score


def pipeCV(pipe, train_data, train_labels, n_cv):
    cv_results = cross_validate(pipe, train_data, train_labels, scoring='roc_auc', return_train_score = True, return_estimator = True, n_jobs = -1)
    return cv_results

def print_elapsed_time(starttime):
    time_now=time.perf_counter()
    elapsed_time=time_now-starttime
    print("Time elapsed since start: %.2f s" % elapsed_time)
    
###############################################################################
    
def solveSubtask1(train_data_imputed_reduced_pd, test_data_imputed_reduced_pd, train_labels_task1_pd):
    # Inputs: Imputed Data Sets with ONLY 1 row per patient!
    
    totaltime_start=time.perf_counter()
    # Solves Subtask 1 of Task 2 Project IML
    print("TASK 1: Fit Multilabel Classifier starts. Set up Pipeline and Cross Validate with MLPClassifier (solver= adam, activation= logistic, max_iter = 150) and n_cv = 5")
    print_elapsed_time(totaltime_start)
    
    train_data_imputed = train_data_imputed_reduced_pd.values
    test_data_imputed = test_data_imputed_reduced_pd.values
    train_labels = train_labels_task1_pd.values
    
    ## Binary Classification for multiple labels with scoring function AUC
    # SVC works a bit less well than MLPClassifier and takes longer to train.
    #pipe_1_svc = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(class_weight = 'balanced', random_state = 42), n_jobs = -1))
    pipe_1_mlp = make_pipeline(StandardScaler(), MLPClassifier(solver = 'adam', activation='logistic', random_state=42, max_iter = 150))
    
    #result_1_svc = pipeCV(pipe_1_svc, train_data_reduced_withGrad[:, 2:], Y1[:, 1:], 5)
    result_1_mlp = pipeCV(pipe_1_mlp, train_data_imputed[:, 2:], train_labels, 5)   
    
    print("TASK 1: Crossvalidation complete. Fit Estimator")
    print_elapsed_time(totaltime_start)
    
    mlp_test_score = result_1_mlp['test_score']
    avg_test_score = np.mean(mlp_test_score)
    print(f'Average Cross Validation ROC AUC Score: {avg_test_score}')
    pipe_1_mlp.fit(train_data_imputed[:, 2:], train_labels)
        
    print("TASK 1: Fit complete, calculate Prediction Probability Output")
    print_elapsed_time(totaltime_start)
    task_1_output = pipe_1_mlp.predict_proba(test_data_imputed[:, 2:])
    task_1_output_pd = pd.DataFrame(task_1_output, columns = train_labels_task1_pd.columns)
    
    return task_1_output_pd
    

###############################################################################
    