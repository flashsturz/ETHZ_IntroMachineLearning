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
# 6) verkleinere Grid,
#
################################

#['pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos',\
#'LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2',\
#'LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate',\
#'LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']


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

#def print_df_shema(df1, df2, df3):
#    print('sizes: \n',
#          "train features: \t\t", df1.shape, "\n", type(df1),
#          "train labels (sepsis): ", df2.shape, "\n", type(df2),
#          "test features: \t\t", df3.shape, type(df3))

working_dir_path = '/Users/janikbaumer/Documents/Studium/Master_2_sem/IML/' \
                   'Project/zueriost/ETHZ_IntroMachineLearning/Task 2/Data_2_new'

# define files
file_train_features_imputed = 'train_features_imp.csv'
file_train_features_imputed_reduced = 'train_features_reduced.csv'
file_train_features_imputed_reduced_grad = 'train_features_reduced_withGrad.csv'
file_train_labels = 'train_labels.csv'
file_test_features_imputed = 'test_features_imp.csv' # ToDo: get reduced file


# setup
setup(working_dir_path)

# reading data
print(f"Start reading data: \n Time elapsed: {datetime.now() - start}")
df_train_features = get_features(file_train_features_imputed_reduced)
df_test_features = get_features(file_test_features_imputed)
df_train_labels_sepsis = get_train_label_sepsis(file_train_labels)
print(f"Done reading data: \n Time elapsed: {datetime.now() - start} \n")



print('sizes: \n',
      "train features: \t\t", df_train_features.shape, type(df_train_features), "\n",
      "train labels (sepsis) : \t", df_train_labels_sepsis.shape, type(df_train_labels_sepsis), "\n",
      "test features: \t\t", df_test_features.shape, type(df_test_features))
print()



#print(f"df train features: \n, {df_train_features}")
#print()
#print(f"df train labels sepsis: \n{df_train_labels_sepsis}")

# Scaling:
print(f"Start Scaling data: \n Time elapsed: {datetime.now() - start}")
# get scaled features
X_train = scale_data(df_train_features)
X_test = scale_data(df_test_features)
# get labels (non-scaled)
Y_train = df_train_labels_sepsis.to_numpy()
#print("Y BEFORE RAVEL: \n", Y_train[0:20, :])
Y_train =  Y_train.ravel()
#print("Y AFTER RAVEL: \n", Y_train[0:20])
print(f"Done Scaling data: \n Time elapsed: {datetime.now() - start} \n")




print(f"df train features after scaling: \n, {df_train_features}")
print()
print(f"df train labels sepsis after scaling: \n{df_train_labels_sepsis}")

##### X_train_temp, X_validation, Y_train_temp, Y_validation = train_test_split(X_train_prep, Y_train, test_size=0.2)

# uniques = df_train_labels_sepsis.LABEL_Sepsis.unique()
# print(f"unique values: {uniques}")
# print("how many ones: ", df_train_labels_sepsis[df_train_labels_sepsis.LABEL_Sepsis == 1].shape[0])
# print("how many zeroes: ", df_train_labels_sepsis[df_train_labels_sepsis.LABEL_Sepsis == 0].shape[0])


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


param_grid_SVC = {
    'C' : [0.01],
    'kernel' : ['linear'],
    'degree' : [3],
    'probability' : [True]
}

param_grid_LogRegr = {
    'penalty' : ['l2'],
    'solver' : ['liblinear', 'newton-cg'],
    'max_iter' : np.linspace(100, 200, 3), #[50000, 70000, 1000000, 120000, 150000],
    'C' : [0.1, 1, 10]
}

param_grid_forest = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [2, 4, 6, 8, 10, 12]
}


model_svc = SVC()
model_LR = LogisticRegression()
model_forest = DecisionTreeClassifier()
2
# grid = GridSearchCV(estimator= model_forest, param_grid = param_grid_forest, scoring='roc_auc') #, scoring= roc_auc_score)
grid = GridSearchCV(estimator= model_LR, param_grid = param_grid_LogRegr, scoring='roc_auc', n_jobs=2) #, scoring= roc_auc_score)
# grid = GridSearchCV(estimator= model_svc, param_grid = param_grid_SVC, scoring='roc_auc') #, scoring= roc_auc_score)

print(f"grid: \n {grid}")

print(f"Start fitting Grid Search CV: \n Time elapsed: {datetime.now() - start}")
print("Fitting...")
grid.fit(X_train, Y_train) # todo: check if id in X_train - ev delete
print(f"Done fitting Grid Search CV: \n Time elapsed: {datetime.now() - start}")

print()
print("Grid Search Best Score: \n", grid.best_score_)
print("Grid Search Best Estimator: \n", grid.best_estimator_)
#print("Grid Search Best Estimator: \n", grid.best_estimator_)
print()


print("Grid Search all CV Results: \n", grid.cv_results_)









Y_pred_decfct = grid.decision_function(X_train)
Y_pred_decfct = sigmoid(Y_pred_decfct)
Y_pred = grid.predict(X_train)

print("type y_pred_decfct after sigmoid: ", type(Y_pred_decfct))
print(Y_pred_decfct[0:20])
print()
print("type y pred: ", type(Y_pred))
print(Y_pred[0:20])



"""
print("Y pred prob after predict proba: ", Y_pred_prob.shape)

print("first elements of Y pred prob after sigmoid: \n", Y_pred_prob[0:20, :])

print("Y pred after predict: ", Y_pred.shape)
print("first elements of Y pred: ", Y_pred[0:20])
"""

#task_2_output = pipe_1_mlp.predict_proba(test_data_reduced_withGrad[:, 2:])

# np.savetxt('../Data_2/Task_2_Subtask_2_Predictions.csv', task_2_output, fmt='%.3f', delimiter = ',', header = 'LABEL_Sepsis', comments='')























"""
print()
print("list of col names (train features): ")
print(list(df_train_features.columns))
print()
print("list of col names (test features): ")
print(list(df_test_features.columns))
print(df_train_labels_sepsis_repeated.head())
"""


"""


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)
Y_test = clf.predict(X_test)

print("y test type", Y_test.shape)
print()
print("y test: \n", Y_test)
"""

"""
#SVC
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print("type of y predict: \n", type(y_pred))
print(f"y predict: \n {y_pred}")
dec_fct_output = clf.decision_function(X_test)
print("type of dec fct output: \n", type(dec_fct_output))
print(f"dec fct output: \n {dec_fct_output}")


#SVM = svm.LinearSVC()
#SVM.fit(df_train_features, df_train_labels_sepsis_repeated)
#y_pred_SVM = SVM.predict(df_test_features)
"""


"""
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print("type of y predict: \n", y_pred)
"""


"""
LR does not converge
LR = LogisticRegression(random_state=42, solver='sag', multi_class='ovr').fit(df_train_features, df_train_labels_sepsis_repeated)
y_pred_LR = LR.predict(df_test_features)
print("type y pred LR: ", type(y_pred_LR), " / ", "and length: ", len(y_pred_LR))
print("y pred LR: \n", y_pred_LR)
"""


"""
dectree = tree.DecisionTreeClassifier()
dectree = dectree.fit(df_train_features, df_train_labels_sepsis_repeated)
y_pred_dectree = dectree.predict(df_test_features)
print("type y pred SVM: ", type(y_pred_dectree), " / ", "and length: ", len(y_pred_dectree))
print("y pred Decision Tree: \n", y_pred_dectree)
"""




"""
# next steps
# loop over 12 elements, compromise to one single value (if more 1 -> 1, else 0)
# so that we have one value per Versuchsperson


#task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], y_pred_dectree['LABEL_Sepsis'])

print(f"execution time: {datetime.now() - start}")

x_train_2, y_train_2 =
x_train_2, x_validate_2, y_train_2, y_validate_2 = train_test_split(train_data_reduced_withGrad, Y2, test_size = 0.2)


# from Flavio:
# Binary Classification for Sepsis Risk
clf_2 = make_pipeline(StandardScaler(), SVC(gamma='auto', class_weight = 'balanced', random_state = 42, probability = True))
clf_2.fit(x_train_2[:, 2:], y_train_2[:, 1])
y_pred_val_2 = clf_2.predict(x_val_2[:,2:])
y_pred_val_2_proba = clf_2.predict_proba(x_val_2[:, 2:])
sigmoid_2 = 1/(1 + np.exp(-clf_2.decision_function(x_val_2[:,2:])))

score_pred = roc_auc_score(y_val_2[:, 1], y_pred_val)
score_proba = roc_auc_score(y_val_2[:, 1], y_pred_val_proba[:, 1])
score_dec_func = roc_auc_score(y_val_2[:, 1], sigmoid_2)
"""


print(f"Done full script: \n Time elapsed: {datetime.now() - start}")
