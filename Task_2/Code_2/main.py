# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Goal of this task is to first impute the missing values in the data set and then perform two different binary classification tasks and one regression task.
#--------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from scipy import stats
import time

from sklearn.pipeline import make_pipeline

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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

def print_elapsed_time(starttime):
    time_now=time.perf_counter()
    elapsed_time=time_now-starttime
    print("Time elapsed since start: %.2f s" % elapsed_time)
    
# ------------- Main ----------------
# toggle data import
data_import_active = True
# use gradient data
use_gradients = True
# read in Data with different Impute Methods
use_different_imputer = True

print(f'Starts. Data Import = {data_import_active}, Use Gradients = {use_gradients}')
totaltime_start=time.perf_counter()

if use_different_imputer:
    train_data_reduced_simon_mean = np.genfromtxt('../Data_2/Different_Imputation_Methods/X_MAT_train_features_simpleIMP_mean_12h_on1line.csv', dtype=float, delimiter=',')
    train_data_reduced_simon_median = np.genfromtxt('../Data_2/Different_Imputation_Methods/X_MAT_train_features_simpleIMP_median_12h_on1line.csv', dtype=float, delimiter=',')
    train_data_reduced_simon_constant = np.genfromtxt('../Data_2/Different_Imputation_Methods/X_MAT_train_simpleIMP_constant_12h_on1line.csv', dtype=float, delimiter=',')
    
    test_data_reduced_simon_mean = np.genfromtxt('../Data_2/Different_Imputation_Methods/X_MAT_test_simpleIMP_mean_12h_on1line.csv', dtype=float, delimiter=',')
    test_data_reduced_simon_median = np.genfromtxt('../Data_2/Different_Imputation_Methods/X_MAT_test_simpleIMP_median_12h_on1line.csv', dtype=float, delimiter=',')
    test_data_reduced_simon_constant = np.genfromtxt('../Data_2/Different_Imputation_Methods/X_MAT_test_simpleIMP_constant_12h_on1line.csv', dtype=float, delimiter=',')

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
    print("Finished Preparation")
    print_elapsed_time(totaltime_start)
    
    if not os.path.isfile('../Data_2/train_features_imp.csv'):
        print("Since no imputed Data found, calculate Imputed Dataset...")
        imputer = IterativeImputer(random_state = 42)
        train_data_imp = imputer.fit_transform(train_data)
        test_data_imp = imputer.transform(test_data) # Transform test_data with same rules as train data
        np.savetxt('../Data_2/train_features_imp.csv', train_data_imp, fmt=('%.3f'), delimiter=',', comments='',)
        np.savetxt('../Data_2/test_features_imp.csv', test_data_imp, fmt=('%.3f'), delimiter=',', comments='',)
    if os.path.isfile('../Data_2/train_features_imp.csv'):
        print("Imputed Data Set found, read in File")
        train_data_imp = np.genfromtxt('../Data_2/train_features_imp.csv', dtype=float, delimiter=',')
        test_data_imp = np.genfromtxt('../Data_2/test_features_imp.csv', dtype=float, delimiter=',')
    print("Finished Imputation. Reducing Data Starts")
    print_elapsed_time(totaltime_start)
    # print('Missing after imputation: %d' % sum(np.isnan(train_data_imp).flatten()))
    if not os.path.isfile('../Data_2/train_features_reduced.csv'):
        print("No Reduced Data Set found, compute reduced set...")
        train_data_reduced, test_data_reduced = combineHourlyMeasurementsPerPatient(train_data_imp, test_data_imp)
        np.savetxt('../Data_2/train_features_reduced.csv', train_data_reduced, fmt=('%.3f'), delimiter=',', comments='',)
        np.savetxt('../Data_2/test_features_reduced.csv', test_data_reduced, fmt=('%.3f'), delimiter=',', comments='',)
    else:
        print("Reduced Data Set found, read in Data...")
        train_data_reduced = np.genfromtxt('../Data_2/train_features_reduced.csv', dtype=float, delimiter=',')
        test_data_reduced = np.genfromtxt('../Data_2/test_features_reduced.csv', dtype=float, delimiter=',')
    print("Finished Reduction of Data.")
    print_elapsed_time(totaltime_start)
    # Create Data Arrays for Subtasks
    iy_1 = [i for i in range(train_labels.shape[1]) if i not in [11, 12, 13, 14, 15]]
    iy_2 = [i for i in range(train_labels.shape[1]) if i in [0, 11]]
    iy_3 = [i for i in range(train_labels.shape[1]) if i not in range(1,12)]
    Y1, Y2, Y3 = train_labels[:, iy_1], train_labels[:, iy_2], train_labels[:, iy_3]

if use_gradients:
    print("Gradients:")
    if not os.path.isfile('../Data_2/train_features_reduced_withGrad.csv'):
        print("No Gradient Set found, calculating Gradients...")
        train_gradients = FeatureGradients(train_data_imp, 'train')
        test_gradients = FeatureGradients(test_data_imp, 'test')
        train_data_reduced_withGrad = np.c_[train_data_reduced, train_gradients]
        test_data_reduced_withGrad = np.c_[test_data_reduced, test_gradients]
        np.savetxt('../Data_2/train_features_reduced_withGrad.csv', train_data_reduced_withGrad, fmt=('%.3f'), delimiter=',', comments='',)
        np.savetxt('../Data_2/test_features_reduced_withGrad.csv', test_data_reduced_withGrad, fmt=('%.3f'), delimiter=',', comments='',)
    else:
        print("Gradient Data Set found, read in Data...")
        train_data_reduced_withGrad = np.genfromtxt('../Data_2/train_features_reduced_withGrad.csv', dtype=float, delimiter=',')
        test_data_reduced_withGrad = np.genfromtxt('../Data_2/test_features_reduced_withGrad.csv', dtype=float, delimiter=',')
    print("Finished Gradients.")
    print_elapsed_time(totaltime_start)

# Model Data Preparation
x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(train_data_reduced_withGrad, Y1, test_size = 0.2)
x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(train_data_reduced_withGrad, Y2, test_size = 0.2)
x_train_3, x_val_3, y_train_3, y_val_3 = train_test_split(train_data_reduced_withGrad, Y3, test_size = 0.2)

###### TASK 1 ########
print("TASK 1: Fit Multilabel Classifier starts. Set up Pipeline and Cross Validate with n_cv = 5")
print_elapsed_time(totaltime_start)
## Binary Classification for multiple labels with scoring function AUC
# SVC works a bit less well than MLPClassifier and takes longer to train.
#pipe_1_svc = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(class_weight = 'balanced', random_state = 42), n_jobs = -1))
pipe_1_mlp = make_pipeline(StandardScaler(), MLPClassifier(solver = 'adam', activation='logistic', random_state=42, max_iter = 150))

#result_1_svc = pipeCV(pipe_1_svc, train_data_reduced_withGrad[:, 2:], Y1[:, 1:], 5)
result_1_mlp = pipeCV(pipe_1_mlp, train_data_reduced_withGrad[:, 2:], Y1[:, 1:], 5)

if use_different_imputer:
    print("TASK 1: Testing with different imputation methods")
    print("Approaches Simon: Simple Impute (Mean, Median, Const.)")
    print_elapsed_time(totaltime_start)
    
    pipe_1_mlp_simon = make_pipeline(StandardScaler(), MLPClassifier(solver = 'adam', activation='logistic', random_state=42, max_iter = 150))
    pipe_1_mlp_simon_nonStandardized = make_pipeline(MLPClassifier(solver = 'adam', activation='logistic', random_state=42, max_iter = 150))
    
    print("Cross Validation with Imputation method: MEAN")
    print_elapsed_time(totaltime_start)
    result_1_mlp_simon_mean = pipeCV(pipe_1_mlp_simon, train_data_reduced_simon_mean, Y1[:, 1:], 5)
    result_1_mlp_simon_mean_nonStandardized = pipeCV(pipe_1_mlp_simon_nonStandardized, train_data_reduced_simon_mean, Y1[:, 1:], 5)
    simon_mean_test_score_average = np.mean(result_1_mlp_simon_mean['test_score'])
    simon_mean_nonStandardized_test_score_average = np.mean(result_1_mlp_simon_mean_nonStandardized['test_score'])
    
    print("Cross Validation with Imputation method: MEDIAN")
    print_elapsed_time(totaltime_start)
    result_1_mlp_simon_median = pipeCV(pipe_1_mlp_simon, train_data_reduced_simon_median, Y1[:, 1:], 5)
    result_1_mlp_simon_median_nonStandardized = pipeCV(pipe_1_mlp_simon_nonStandardized, train_data_reduced_simon_median, Y1[:, 1:], 5)
    simon_median_test_score_average = np.mean(result_1_mlp_simon_median['test_score'])
    simon_median_nonStandardized_test_score_average = np.mean(result_1_mlp_simon_median_nonStandardized['test_score'])
    
    print("Cross Validation with Imputation method: CONSTANT")
    print_elapsed_time(totaltime_start)
    result_1_mlp_simon_constant = pipeCV(pipe_1_mlp_simon, train_data_reduced_simon_constant, Y1[:, 1:], 5)
    result_1_mlp_simon_constant_nonStandardized = pipeCV(pipe_1_mlp_simon_nonStandardized, train_data_reduced_simon_constant, Y1[:, 1:], 5)
    simon_constant_test_score_average = np.mean(result_1_mlp_simon_constant['test_score'])
    simon_constant_nonStandardized_test_score_average = np.mean(result_1_mlp_simon_constant_nonStandardized['test_score'])
    

print("TASK 1: Crossvalidation complete. Fit Estimator")
print_elapsed_time(totaltime_start)

mlp_test_score = result_1_mlp['test_score']
avg_test_score = np.mean(mlp_test_score)
pipe_1_mlp.fit(train_data_reduced_withGrad[:, 2:], Y1[:, 1:])

if use_different_imputer:
    print("Fitting Simon Data")
    print_elapsed_time(totaltime_start)
    pipe_1_mlp_simon.fit(train_data_reduced_simon_constant, Y1[:, 1:])
    pipe_1_mlp_simon_nonStandardized.fit(train_data_reduced_simon_constant, Y1[:, 1:])
    
    task_1_output_simon_const = pipe_1_mlp_simon.predict_proba(test_data_reduced_simon_constant)
    task_1_output_simon_const_nonStandardized = pipe_1_mlp_simon_nonStandardized.predict_proba(test_data_reduced_simon_constant)
    
    np.savetxt('../Data_2/Different_Imputation_Methods/Prediction_simon_fitConst.csv', task_1_output_simon_const, fmt='%.5f', delimiter = ',', header = 'LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2', comments='')
    np.savetxt('../Data_2/Different_Imputation_Methods/Prediction_simon_fitConst_nonStandardized.csv', task_1_output_simon_const_nonStandardized, fmt='%.5f', delimiter = ',', header = 'LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2', comments='')
    
    pipe_1_mlp_simon.fit(train_data_reduced_simon_mean, Y1[:, 1:])
    pipe_1_mlp_simon_nonStandardized.fit(train_data_reduced_simon_mean, Y1[:, 1:])
    
    task_1_output_simon_mean = pipe_1_mlp_simon.predict_proba(test_data_reduced_simon_constant)
    task_1_output_simon_mean_nonStandardized = pipe_1_mlp_simon_nonStandardized.predict_proba(test_data_reduced_simon_constant)
    
    np.savetxt('../Data_2/Different_Imputation_Methods/Prediction_simon_fitMean.csv', task_1_output_simon_mean, fmt='%.5f', delimiter = ',', header = 'LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2', comments='')
    np.savetxt('../Data_2/Different_Imputation_Methods/Prediction_simon_fitMean_nonStandardized.csv', task_1_output_simon_mean_nonStandardized, fmt='%.5f', delimiter = ',', header = 'LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2', comments='')

    
print("TASK 1: Fit complete, calculate Prediction Probability Output")
print_elapsed_time(totaltime_start)
task_1_output = pipe_1_mlp.predict_proba(test_data_reduced_withGrad[:, 2:])

np.savetxt('../Data_2/Task_2_Subtask_1_Predictions.csv', task_1_output, fmt='%.5f', delimiter = ',', header = 'LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2', comments='')
# print(f'Task 1 Validation ROC AUC Score: {roc_auc_score(y_val_1[:, 1:], pipe_1_mlp.predict_proba(x_val_1[:, 2:]))}')

########################## GridSearch CV #################
# Best Estimator Combination according to GridSearch:
# No Dimension Reduction, normalize, MLP Classifier with adam, max_iter = 400
# Best Parameter for SVC --> C = 1.0

#pipe = Pipeline([
#        ('normalize', 'passthrough'),
#        ('reduce_dim', 'passthrough'),
#        ('clf', 'passthrough')])
#    
#param_grid = [
#        ######### PCA
#        {
#                'normalize': [StandardScaler()], 
#                'reduce_dim': [PCA(iterated_power = 2, random_state = 42)],
#                'reduce_dim__n_components': [50, 60, 69, 'mle'],
#                'clf': [OneVsRestClassifier(LogisticRegression(fit_intercept = True, class_weight = 'balanced', multi_class = 'ovr', random_state = 42, max_iter = 200), n_jobs = -1)],
#                'clf__estimator__C': [1.0, 10.0, 100.0],
#                'clf__estimator__solver': ['saga', 'liblinear'],
#                },
#        {
#                'normalize': [StandardScaler()],
#                'reduce_dim': [PCA(iterated_power = 2, random_state = 42)],
#                #'reduce_dim__n_components': [8, 9, 10],
#                'reduce_dim__n_components': [50, 60, 69, 'mle'],
#                'clf': [MLPClassifier(solver = 'adam', activation='logistic', random_state=42, max_iter = 400)]
#                },
#        {
#                'normalize': [StandardScaler()],
#                'reduce_dim': [PCA(iterated_power = 2, random_state = 42)],
#                'reduce_dim__n_components': [50, 60, 69, 'mle'],
#                'clf': [OneVsRestClassifier(LinearSVC(class_weight = 'balanced', random_state = 42, max_iter = 1500), n_jobs = -1), 
#                                                 OneVsRestClassifier(SVC(class_weight = 'balanced', random_state = 42), n_jobs = -1)],
#                'clf__estimator__C': [0.1, 1.0, 10.0, 100.0],
#                },
#        {
#                'normalize': [StandardScaler()],
#                'reduce_dim': [PCA(iterated_power = 2, random_state = 42)],
#                'reduce_dim__n_components': [50, 60, 69, 'mle'],
#                'clf': [RandomForestClassifier(n_jobs = -1, random_state = 42, class_weight = 'balanced_subsample' ), 
#                                                 RandomForestClassifier(n_jobs = -1, random_state = 42, class_weight = 'balanced' )],
#                'clf__max_depth': [2, 6, 10],
#                'clf__n_estimators': [10, 25, 50, 100]
#                }
#    ]
#
#
#timestr = time.strftime("%Y%m%d_%H%M%S")
#grid = GridSearchCV(pipe, n_jobs=-1, param_grid = param_grid, scoring='roc_auc', cv=3, verbose = 1)
#
#grid.fit(train_data_reduced_withGrad[:, 2:], Y1[:, 1:])
#results_dataframe = pd.DataFrame.from_dict(grid.cv_results_)
#results_dataframe.to_csv(f'../Data_2/GridSearchCV_{timestr}.csv')
#mean_cv_scores_1 = np.array(grid.cv_results_['mean_test_score'])
#######################     End Grid Search CV      #############################################

print("TASK 1: Finished")
print_elapsed_time(totaltime_start)


###### TASK 2 ########
# Binary Classification for Sepsis Risk
clf_2 = make_pipeline(StandardScaler(), SVC(gamma='auto', class_weight = 'balanced', random_state = 42, probability = True))
clf_2.fit(x_train_2[:, 2:], y_train_2[:, 1])
y_pred_val_2 = clf_2.predict(x_val_2[:,2:])
confidence_score_2 = clf_2.decision_function(x_val_2[:,2:])
sigmoid_2 = 1/(1 + np.exp(-clf_2.decision_function(x_val_2[:,2:])))
#
#score_pred = roc_auc_score(y_val_2[:, 1], y_pred_val_2)
#score_dec_func = roc_auc_score(y_val_2[:, 1], sigmoid_2)


###### TASK 3 ########

