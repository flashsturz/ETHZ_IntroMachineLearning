# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
# -----------------
# Description Task 2

# -------------------------------------------------------------------------------------------------
# IMPORT
# ------

import pandas as pd
import numpy as np
import random
import FeatureTransform_simpleImp
import FeatureTransform_IterativeImp
import Subtask1
import Subtask2
import Subtask3
from datetime import datetime
import time as time


# -------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------

def print_elapsed_time(starttime):
    time_now = time.perf_counter()
    elapsed_time = time_now-starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)


# -------------------------------------------------------------------------------------------------
# VARIABLES
# ---------

PATH_TRAIN_FEATURES = 'Data_2/train_features_SHORT_FOR_TESTING.csv'
PATH_TEST_FEATURES = 'Data_2/test_features_SHORT_FOR_TESTING.csv'
PATH_TRAIN_LABELS = 'Data_2/train_labels_SHORT_FOR_TESTING.csv'
PATH_SAMPLE_FILE = 'Data_2/sample.csv'

COL_SUBTASK1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
                'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
COL_SUBTASK2 = ['LABEL_Sepsis']
COL_SUBTASK3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

COL_ALL = ['pid']+COL_SUBTASK1+COL_SUBTASK2+COL_SUBTASK3


# -------------------------------------------------------------------------------------------------
# PREPS
# -----

print('=====Execution starts.======')
print('=====   Preparations...')

random.seed(1234)

current_time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
train_features_pd = pd.read_csv(PATH_TRAIN_FEATURES)
test_features_pd = pd.read_csv(PATH_TEST_FEATURES)
train_labels_pd = pd.read_csv(PATH_TRAIN_LABELS)

sample_pd = pd.read_csv(PATH_SAMPLE_FILE)

gradients_active = 1  # use of Gradients in Iterative Imputer
gradients_inactive = False  # use of Gradients in Iterative Imputer

list_pid = test_features_pd.pid.unique()  # get list of all pid in test_data

# -------------------------------------------------------------------------------------------------
# Run imputer functions provided in imputer files.
#    Many pandas dataframes with imputed feature data.

print('=====   Preparations finished. Imputing...')

# Get simpleImpute data
[test_imp_constant_pd, train_imp_constant_pd] = FeatureTransform_simpleImp.simpleimp_constant(test_features_pd, train_features_pd)
[test_imp_mean_pd, train_imp_mean_pd] = FeatureTransform_simpleImp.simpleimp_mean(test_features_pd, train_features_pd)
[test_imp_median_pd, train_imp_median_pd] = FeatureTransform_simpleImp.simpleimp_median(test_features_pd, train_features_pd)


# Get IterativeImpute data
train_data_reduced_pd, test_data_reduced_pd, train_data_imp_pd, test_data_imp_pd = FeatureTransform_IterativeImp.iterativeImpute(PATH_TRAIN_FEATURES, PATH_TEST_FEATURES, gradients_inactive)
train_data_reduced_withGrad_pd, test_data_reduced_withGrad_pd, train_data_imp_pd, test_data_imp_pd = FeatureTransform_IterativeImp.iterativeImpute(PATH_TRAIN_FEATURES, PATH_TEST_FEATURES, gradients_active)


# -------------------------------------------------------------------------------------------------
# Run subtask1

print('=====   Imputing finished. Solving subtask1...')

# Extract Train Labels for Subtask 1
train_labels_task1_pd = train_labels_pd[['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']]

# Calculate Output for Subtask 1
result_subtask1_pd = Subtask1.solveSubtask1(train_data_reduced_withGrad_pd, test_data_reduced_withGrad_pd, train_labels_task1_pd)


# -------------------------------------------------------------------------------------------------
# Run subtask2

print('=====   Subtask1 finished. Solving subtask2...')

train_label_subtask2 = train_labels_pd['LABEL_Sepsis']
train_data_subtask2 = train_data_reduced_pd
del train_data_subtask2['pid']
test_data_subtask2 = test_data_reduced_pd
del test_data_subtask2['pid']

result_subtask2_pd = Subtask2.solveSubtask2(train_data_reduced_pd, train_label_subtask2, test_data_reduced_pd)


# -------------------------------------------------------------------------------------------------
# Run subtask3

print('=====   Subtask2 finished. Solving subtask3...')

result_subtask3_pd = Subtask3.solveSubtask3(train_imp_mean_pd, test_imp_mean_pd, train_labels_pd, current_time_str, verbose=1)


# -------------------------------------------------------------------------------------------------
# Combine Result files

print('=====   Subtask3 finished. Results getting combined and written to file...')

full_submission_pd = pd.DataFrame(data=np.zeros((len(list_pid), len(sample_pd.columns))), columns=sample_pd.columns)
full_submission_pd['pid'] = np.array(list_pid).T

full_submission_pd[COL_SUBTASK1] = result_subtask1_pd[COL_SUBTASK1]
full_submission_pd[COL_SUBTASK2] = result_subtask2_pd[COL_SUBTASK2]
full_submission_pd[COL_SUBTASK3] = result_subtask3_pd[COL_SUBTASK3]


# -------------------------------------------------------------------------------------------------
# Print to File:

# .csv-file:
pd.DataFrame(full_submission_pd).to_csv("full_submission" + current_time_str + ".csv", index=False, float_format='%.3f')
# .zip-file
pd.DataFrame(full_submission_pd).to_csv("full_submission" + current_time_str + ".zip", index=False,
                                        float_format='%.3f',
                                        compression='zip')

print('=====   Finished. The results are written to the following files: ')
print('=====        full_submission_<current_timestamp>.csv')
print('=====        full_submission_<current_timestamp>.zip (File to be submitted).')
