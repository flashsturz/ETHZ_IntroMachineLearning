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
import pandas as pd
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
# from score_submission import get_score
start = datetime.now()

################################
# Aufbau Code (Vorgehen):
#
# 1a) einlesen von trainings und test daten (bereits imputed, also ohne NaN values)
#
# 1b) Daten von 12h auf 1h runterbrechen
#
# 2) trainingsdaten normalisieren und standardisieren (mit StandardScaler)
#
# 3) entscheide für einen Estimator (z.B. SVM, LR, DecisionTreeClassifier etc)
#
# 4) GridSearchCV über diesen Estimator mit geeigneten Werten
#
# 5) schreibe grid.cv_results, grid.best_estimator_ / grid.best_score_ / grid.best_params / grid.best_index / grid.scorer_ / grid.n_splits_
#    in ein file
#
# 6) verkleinere Grid, optimieren
#
################################

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
    #print("scaled dataframe (printed in fct): \n", scaled)
    return scaled

def sigmoid(x):
 return 1/(1 + np.exp(-x))

estim = 'SVC' # set either to LR, SVC, forest
scaling = False # set either true or false

# define path
working_dir_path = '../Data_2_new'

# define files
file_train_features_imputed = 'train_features_imp.csv'
file_train_features_imputed_reduced = 'train_features_reduced.csv'
file_train_features_imputed_reduced_grad = 'train_features_reduced_withGrad.csv'
file_train_labels = 'train_labels.csv'
file_test_features_imputed = 'test_features_imp.csv' # ToDo: get reduced file
file_to_write = 'Task_2_Subtask_2_Predictions.csv'

# setup
setup(working_dir_path)

# reading data
print(f"Start reading data: \n Time elapsed: {datetime.now() - start}")
df_train_features = get_features(file_train_features_imputed_reduced)
df_test_features = get_features(file_test_features_imputed)
df_train_labels_sepsis = get_train_label_sepsis(file_train_labels)
print(f"Done reading data: \n Time elapsed: {datetime.now() - start} \n")


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


#print(f"df train features after scaling: \n, {df_train_features}")
#print()
#print(f"df train labels sepsis after scaling: \n{df_train_labels_sepsis}")

# uniques = df_train_labels_sepsis.LABEL_Sepsis.unique()
# print(f"unique values: {uniques}")
# print("number of ones: ", df_train_labels_sepsis[df_train_labels_sepsis.LABEL_Sepsis == 1].shape[0])
# print("number of zeroes: ", df_train_labels_sepsis[df_train_labels_sepsis.LABEL_Sepsis == 0].shape[0])

# print shapes and types
print()
print('sizes: \n',
      "X train: \t", X_train.shape, type(X_train), "\n",
      "X test: \t", X_test.shape, type(X_test), "\n",
      "Y train: \t", Y_train.shape, type(Y_train))

# start with GridSearchCV for
#param_grid = {
#    'C' : [0.01, 0.1, 1, 10],
#    'kernel' : ['linear', 'poly', 'rbf'],
#    'degree' : [2, 3]
#}







# todo: set estimator variable, choose grid depending on variable

if estim == 'forest':
    param_grid_forest = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 6, 8, 10, 12]
    }
    model_forest = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(estimator= model_forest, param_grid = param_grid_forest, scoring='roc_auc') #, scoring= roc_auc_score)

elif estim == 'LR':
    param_grid_LogRegr = {
        'penalty': ['l2'],
        'solver': ['liblinear', 'newton-cg'],
        'max_iter': np.linspace(10, 200, 5),  # [50000, 70000, 1000000, 120000, 150000],
        'C': [0.1, 1, 10, 20, 30, 40]
    }
    model_LR = LogisticRegression(random_state=42)
    grid = GridSearchCV(estimator= model_LR, param_grid = param_grid_LogRegr, scoring='roc_auc', n_jobs=-1) #, scoring= roc_auc_score)

elif estim == 'SVC':
    param_grid_SVC = {
        'C': [10, 20, 40],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3],
        'probability': [True],
        'class_weight' : ['balanced']
    }
    model_svc = SVC(random_state=42)
    grid = GridSearchCV(estimator= model_svc, param_grid = param_grid_SVC, scoring='roc_auc') #, scoring= roc_auc_score)

else:
    print("Variable <estim> has non-allowed value")

# get full grid to be looped over
print(f"grid: \n {grid}")

# Grid Search CV, fitting
print(f"Start fitting Grid Search CV: \n Time elapsed: {datetime.now() - start}")
print("Fitting...")
grid.fit(X_train, Y_train) # todo: check if id in X_train - ev delete
print(f"Done fitting Grid Search CV: \n Time elapsed: {datetime.now() - start}")
print(f"(model used: {estim})")

# Grid Search CV, print Best Score, Best Estimator (and all Results)
print()
print("Grid Search Best Score: \n", grid.best_score_)
print("Grid Search Best Estimator: \n", grid.best_estimator_)
print()
# print("Grid Search all CV Results: \n", grid.cv_results_)


# get Y pred based on X test
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