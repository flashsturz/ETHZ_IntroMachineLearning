# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Description Task 2
# subtask 2: Sepsis Classification
#--------------------------------------------------------------------------------------------------

import numpy as np
import random
import pandas as pd
import sys, os
from sklearn.svm import LinearSVC
import os
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
start = datetime.now()

# definition of functions:
def setup(path):
    os.chdir(path) # os.getcwd() would now get current path (here: 'path')
    np.set_printoptions(threshold=np.inf) # print np.arrays completely

def get_features(path):
    df_features = pd.read_csv(path, sep=',', header = None)
    df_features = df_features.iloc[:, 2:]
    return df_features

def get_train_label_sepsis(path):
    df_train_labels = pd.read_csv(path, sep=',', header = 0)
    #df_train_labels_sepsis = df_train_labels[["pid", "LABEL_Sepsis"]]
    df_train_labels_sepsis = df_train_labels[["LABEL_Sepsis"]]
    return df_train_labels_sepsis

def scale_data(dataframe):
    scaler = StandardScaler() # define standard scaler
    scaled = scaler.fit_transform(dataframe) # transform data
    return scaled

def sigmoid(x):
 return 1/(1 + np.exp(-x))


def solveSubtask2(df_train_X, df_train_Y, df_test_X):
    """
    :param df_train_X: pandas df - without id, only one row per patient
    :param df_train_Y: pandas df (not a Series) - only one column named "LABEL_Sepsis"
    :param df_test_X: pandas df - without id, only one row per patient
    :return: pandas df with probabilities - one per patient, without id, but sorted same as input
    """

    estim = 'forest' # set either to LR, SVC, forest
    scaling = True # set either true or false

    # define path
    working_dir_path = '../Data_2_new'

    # setup
    setup(working_dir_path)

    # define files
    file_train_features_imputed = 'train_features_imp.csv'
    file_train_features_imputed_reduced = 'train_features_reduced_withGrad.csv'
    #file_train_features_imputed_reduced_grad = 'train_features_reduced_withGrad.csv'
    file_train_labels = 'train_labels.csv'
    file_test_features_imputed = 'test_features_reduced_withGrad.csv'
    file_to_write = 'Task_2_Subtask_2_Predictions.csv'

    # reading data
    """
    print(f"Start reading data: \n Time elapsed: {datetime.now() - start}")
    df_train_features = get_features(file_train_features_imputed_reduced)
    df_test_features = get_features(file_test_features_imputed)
    df_train_labels_sepsis = get_train_label_sepsis(file_train_labels)
    print(f"Done reading data: \n Time elapsed: {datetime.now() - start} \n")
    """
    # reading data (for submission):
    df_train_features = df_train_X
    df_test_features = df_train_Y
    df_train_labels_sepsis = df_test_X

    # print shapes and types
    print('sizes: \n',
          "train features: \t\t", df_train_features.shape, type(df_train_features), "\n",
          "train labels (sepsis) : \t", df_train_labels_sepsis.shape, type(df_train_labels_sepsis), "\n",
          "test features: \t\t", df_test_features.shape, type(df_test_features))
    print()


    # Scaling:
    print(f"Start Scaling data: \n Time elapsed: {datetime.now() - start}")

    # get scaled features
    if scaling == True:
        X_train = scale_data(df_train_features)
        X_test = scale_data(df_test_features)
    else:
        X_train = df_train_features.to_numpy()
        X_test = df_test_features.to_numpy()

    # get labels (non-scaled)
    Y_train = df_train_labels_sepsis.to_numpy()
    #print("Y BEFORE RAVEL: \n", Y_train[0:20, :])
    Y_train =  Y_train.ravel()
    #print("Y AFTER RAVEL: \n", Y_train[0:20])
    print(f"Done Scaling data: \n Time elapsed: {datetime.now() - start} \n")

    # print shapes and types
    print()
    print('sizes: \n',
          "X train: \t", X_train.shape, type(X_train), "\n",
          "X test: \t", X_test.shape, type(X_test), "\n",
          "Y train: \t", Y_train.shape, type(Y_train))

    if estim == 'forest':
        param_grid_forest = {
            'criterion': ['gini', 'entropy'], #['gini'],
            'max_depth': [2, 4, 6, 8, 10, 12], #[4]
            'class_weight' : ['balanced', None]
        }
        model_forest = DecisionTreeClassifier(random_state=42)
        grid = GridSearchCV(estimator= model_forest, param_grid = param_grid_forest, scoring='roc_auc') #, scoring= roc_auc_score)


    elif estim == 'LR':

        param_grid_LogRegr = {
            'penalty': ['l1','l2'],
            'solver': ['liblinear', 'newton-cg', 'sag', 'lbfgs'],
            'max_iter': np.linspace(5, 30, 2),  # [50000, 70000, 1000000, 120000, 150000],
            'C': np.linspace(4.6, 6.2, 2), #[0.1, 1, 10, 20, 30, 40]
            'l1_ratio': np.linspace(0.6,0.75,2)
            }
        """
        param_grid_LogRegr = {
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [10, 30, 50, 70],  # [50000, 70000, 1000000, 120000, 150000],
            'C': [1, 10]
        }
        """
        model_LR = LogisticRegression(random_state=42, verbose=False)
        grid = GridSearchCV(estimator= model_LR, param_grid = param_grid_LogRegr, scoring='roc_auc', n_jobs=2) #, scoring= roc_auc_score)

    elif estim == 'SVC':
        param_grid_SVC = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
        ]

        """, #[10, 20, 40],
            'kernel': ('linear', 'rbf'), #['linear', 'rbf', 'poly'],
            'degree': [3], #[2, 3],
            'probability': [True],
            'class_weight' : ['balanced'],
            'gamma' : ['scale', 'auto']
        }
        """
        model_svc = SVC(random_state=42, verbose=True)
        grid = GridSearchCV(estimator=model_svc, param_grid = param_grid_SVC, scoring='roc_auc', n_jobs=2) #, scoring= roc_auc_score)

    else:
        print("Variable <estim> has non-allowed value \n must be either SVC, LR or forest \n actual value:", estim)


    #grid = GridSearchCV(estimator=model_svc, param_grid=param_grid_SVC, scoring='roc_auc', n_jobs=2)


    # get full grid to be looped over
    print(f"grid: \n {grid}")

    print("X train: \n", X_train[0:20,:])
    print()
    print("Y train: \n", Y_train[0:20])
    print()
    print("X test: \n", X_test[0:20,:])


    # Grid Search CV, fitting
    print(f"Start fitting Grid Search CV: \n Time elapsed: {datetime.now() - start}")
    print("Fitting...")
    grid.fit(X_train, Y_train)
    print(f"Done fitting Grid Search CV: \n Time elapsed: {datetime.now() - start}")
    print(f"(model used: {estim})")

    # Grid Search CV, print Best Score, Best Estimator (and all Results)
    print()
    print("Grid Search Best Score: \n", grid.best_score_)
    print("Grid Search Best Estimator: \n", grid.best_estimator_)
    print()
    print("Grid Search all CV Results: \n", grid.cv_results_)

    # print grid.cv_results_ to csv file (via pandas df)
    gscv_results_pd = pd.DataFrame(grid.cv_results_)
    gscv_results_pd.to_csv(str(datetime.now())+'_cv_results.csv')

    # get Y pred based on X test
    if estim == 'forest':
        Y_pred_decfct_sigmoid = grid.predict_proba(X_test)
        Y_pred_decfct_sigmoid = Y_pred_decfct_sigmoid[:,1]
    else:
        Y_pred_decfct = grid.decision_function(X_test)
        Y_pred_decfct_sigmoid = sigmoid(Y_pred_decfct)
    Y_pred = grid.predict(X_test)

    # get type and first elements of actual Y and Y after sigmoid
    print("type y_pred_decfct after sigmoid: ", type(Y_pred_decfct_sigmoid))
    print(Y_pred_decfct_sigmoid[0:20])
    print()
    print("type y pred: ", type(Y_pred))
    print(Y_pred[0:20])

    # writing the Sigmoid Function to file
    task_2_output = Y_pred_decfct_sigmoid
    print('type output: ', type(task_2_output))
    write_time = str(datetime.now())
    np.savetxt(write_time+file_to_write, task_2_output, fmt='%.3f', delimiter=',', header='LABEL_Sepsis', comments='')

    print("Script Execution Time: ", datetime.now()-start)

    # for submission: return df_task_2_output as pandas df instead of ndarray
    df_task_2_output = pd.DataFrame(task_2_output, columns="LABEL_Sepsis")
    print("df task 2 output as pd dataframe: \n", df_task_2_output)
    return df_task_2_output