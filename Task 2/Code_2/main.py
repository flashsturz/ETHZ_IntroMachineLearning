# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Goal of this task is to first impute the missing values in the data set and then perform two different binary classification tasks and one regression task.
#--------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from scipy import stats

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

def FeatureGradients(data_set, data_type):
    # Calculate the rate of change of every feature for every patient as an additional feature
    # Assumes imputed input data without missing values
    gradients = []
    patients = int(train_data.shape[0]/12)
    
    if data_type == 'train':
        N = patients
    if data_type == 'test':
        N = int(data_set.shape[0]/12)
    
    for i in range(0, N):
        data_patient = train_data[12*(i):12*(i+1), :]
        patient_gradient = [0]* (data_set.shape[1]-3) # Since we omit the first three columns in the gradient calculation
        
        for j in range(3, data_set.shape[1]):
            y_vec = data_patient[:, j]
            x_vec = range(0, len(y_vec))
            if y_vec.any():
                if len(y_vec) > 1:    
                    slope, intercept, r_value, p_value, std_er = stats.linregress(x_vec, y_vec)
                    if np.isnan(slope):
                        slope = 0
                else:
                    slope = 0
            else:
                slope = 0
            patient_gradient[j-3] = slope
        gradients.append(patient_gradient)
    return gradients
    
# ------------- Main ----------------
# toggle data import
data_import_active = True
# use gradient data
use_gradients = True


# Read in Data and Processing
if data_import_active or not (os.path.isfile('../Data_2/train_features_imp.csv') and os.path.isfile('../Data_2/train_features_reduced.csv')):
    train_data_frame = pd.read_csv('../Data_2/train_features.csv')
    train_labels_frame = pd.read_csv('../Data_2/train_labels.csv')
    test_data_frame = pd.read_csv('../Data_2/test_features.csv')
    
    train_data = train_data_frame.values
    train_labels = train_labels_frame.values
    test_data = test_data_frame.values
    # print total missing
    # print('Missing before imputation: %d' % sum(np.isnan(train_data).flatten()))
    # Iterative Impute on training data
    if not os.path.isfile('../Data_2/train_features_imp.csv'):
        imputer = IterativeImputer(random_state = 42)
        train_data_imp = imputer.fit_transform(train_data)
        test_data_imp = imputer.transform(test_data) # Transform test_data with same rules as train data
        np.savetxt('../Data_2/train_features_imp.csv', train_data_imp, fmt=('%.3f'), delimiter=',', comments='',)
        np.savetxt('../Data_2/test_features_imp.csv', test_data_imp, fmt=('%.3f'), delimiter=',', comments='',)
    if os.path.isfile('../Data_2/train_features_imp.csv'):
        train_data_imp = np.genfromtxt('../Data_2/train_features_imp.csv', dtype=float, delimiter=',')
        test_data_imp = np.genfromtxt('../Data_2/test_features_imp.csv', dtype=float, delimiter=',')
    # print('Missing after imputation: %d' % sum(np.isnan(train_data_imp).flatten()))
    train_data_reduced, test_data_reduced = combineHourlyMeasurementsPerPatient(train_data_imp, test_data_imp)
    np.savetxt('../Data_2/train_features_reduced.csv', train_data_reduced, fmt=('%.3f'), delimiter=',', comments='',)
    np.savetxt('../Data_2/test_features_reduced.csv', test_data_reduced, fmt=('%.3f'), delimiter=',', comments='',)
    
    # Create Data Arrays for Subtasks
    iy_1 = [i for i in range(train_labels.shape[1]) if i not in [11, 12, 13, 14, 15]]
    iy_2 = [i for i in range(train_labels.shape[1]) if i in [0, 11]]
    iy_3 = [i for i in range(train_labels.shape[1]) if i not in range(1,12)]
    Y1, Y2, Y3 = train_labels[:, iy_1], train_labels[:, iy_2], train_labels[:, iy_3]

if use_gradients:
    train_gradients = FeatureGradients(train_data_imp, 'train')
    test_gradients = FeatureGradients(test_data_imp, 'test')
    train_data_reduced_withGrad = np.c_[train_data_reduced, train_gradients]
    test_data_reduced_withGrad = np.c_[test_data_reduced, test_gradients]


###### TASK 1 ########
# Binary Classification for multiple labels with scoring function AUC


###### TASK 2 ########
# Binary Classification for Sepsis Risk
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(train_data_reduced[:, 2:], Y2[:, 1])
y_pred = clf.predict(test_data_reduced[:,2:])


###### TASK 3 ########

