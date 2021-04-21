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
# from score_submission import get_score
start = datetime.now()
working_dir_path = '/Users/janikbaumer/Documents/Studium/Master_2_sem/IML/' \
                   'Project/zueriost/ETHZ_IntroMachineLearning/Task 2/Data_2'

file_train_features_imputed = 'train_features_imputed_knn_neig2_floatlen3.csv'
file_test_features_imputed = 'test_features_imputed_knn_neig2.csv'
file_train_labels = 'train_labels.csv'


def combineHourlyMeasurementsPerPatient(train_data, test_data):
    # Reduce the data from the 12 hourly measurements per patient to one row per patient using the mean of the 12 entries per feature
    # Assumes imputed input data without missing values
    train_data_comb, test_data_comb = [], []
    patients = int(train_data.shape[0]/12)
    print(f'Patients: {patients}')
    # for training data
    for i in range(0, patients):
        data_patient = train_data[12*(i):12*(i+1), :]
        data_mean = data_patient.mean(axis=0)
        train_data_comb.append(data_mean)
    # for test data
    for i in range(0, int(test_data.shape[0]/12)):
        data_patient = test_data[12*(i):12*(i+1), :]
        data_mean = data_patient.mean(axis=0)
        test_data_comb.append(data_mean)
    return np.asarray(train_data_comb), np.asarray(test_data_comb)


def sigmoid(x):
 return 1/(1 + np.exp(-x))


def setup(path):
    os.chdir(path)
    # os.getcwd() gets current path (here: path)
    np.set_printoptions(threshold=np.inf)



def get_features(path):
    df_features = pd.read_csv(path, sep=',', header=0)
    return df_features

def get_train_label_sepsis_repeated(path):
    df_train_labels = pd.read_csv(path, sep=',', header=0)
    df_train_labels_sepsis = df_train_labels[["pid", "LABEL_Sepsis"]]
    df_train_labels_sepsis_repeated = pd.concat([df_train_labels_sepsis] * 12, ignore_index=True)
    df_train_labels_sepsis_repeated = df_train_labels_sepsis_repeated.sort_values(by=['pid'])
    df_train_labels_sepsis_repeated = df_train_labels_sepsis_repeated["LABEL_Sepsis"]
    df_train_labels_sepsis_repeated = df_train_labels_sepsis_repeated.ravel()
    return df_train_labels_sepsis_repeated


def scale_data(dataframe):
    scaler = StandardScaler() # define standard scaler
    scaled = scaler.fit_transform(dataframe) # transform data
    return scaled



# setup
setup(working_dir_path)

### reading data
df_train_features = get_features(file_train_features_imputed)
df_train_labels_sepsis_repeated = get_train_label_sepsis_repeated(file_train_labels)
df_test_features = get_features(file_test_features_imputed)

print('sizes: \n',
      "train features: ", df_train_features.shape, "\n",
      "train labels (sepsis): ", df_train_labels_sepsis_repeated.shape, "\n",
      "test features: ", df_test_features.shape)

# get scaled features
X_train = scale_data(df_train_features)
X_test = scale_data(df_test_features)

# get labels (non-scaled)
Y_train = df_train_labels_sepsis_repeated



"""
#print shapes
print("train features (rows, cols): ", df_train_features.shape)
print("train labels repeated 12x (rows, cols): ", df_train_labels_sepsis_repeated.shape)
print("test features (rows, cols): ", df_test_features.shape)

print()
print("list of col names (train features): ")
print(list(df_train_features.columns))
print()
print("list of col names (test features): ")
print(list(df_test_features.columns))
print(df_train_labels_sepsis_repeated.head())
"""




"""
#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
#y = np.array([1, 1, 2, 2])

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)
Y_test = clf.predict(X_test)

print("y test type", Y_test.shape)
print()
print("y test: \n", Y_test)
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
