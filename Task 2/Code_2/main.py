# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
# -----------------
# Description Task 2

# INSTRUCTIONS: Run this project from Task 2 folder. Using $python3 Code/main.py
# IMPORTANT: Do not change anything at the files!! The provided files where imputed using the provided imputers.
#            Since the imputing is timeconsuming, they are given as csv-files additionally.

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

PATH_TRAIN_FEATURES = 'Data_2/train_features.csv'
PATH_TEST_FEATURES = 'Data_2/test_features.csv'
PATH_TRAIN_LABELS = 'Data_2/train_labels.csv'
PATH_SAMPLE_FILE = 'Data_2/sample.csv'

COL_SUBTASK1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
                'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
COL_SUBTASK2 = ['LABEL_Sepsis']
COL_SUBTASK3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

COL_ALL = ['pid']+COL_SUBTASK1+COL_SUBTASK2+COL_SUBTASK3

USE_SIMPLEIMP_FILES=False
USE_ITERATIVEIMP_FILES=False

TEST_SIMPLEIMP_MEAN='ImputedFiles/test_features_simpleImpute_mean.csv'
TRAIN_SIMPLEIMP_MEAN='ImputedFiles/train_features_simpleImpute_mean.csv'
TEST_SIMPLEIMP_MEDIAN='ImputedFiles/test_features_simpleImpute_median.csv'
TRAIN_SIMPLEIMP_MEDIAN='ImputedFiles/train_features_simpleImpute_median.csv'
TEST_SIMPLEIMP_CONST='ImputedFiles/test_features_simpleImpute_constant.csv'
TRAIN_SIMPLEIMP_CONST='ImputedFiles/train_features_simpleImpute_constant.csv'
TRAIN_SIMPLEIMP_CONST_REDUCED = 'ImputedFiles/X_MAT_train_simpleIMP_constant_12h_on1line.csv'
TEST_SIMPLEIMP_CONST_REDUCED = 'ImputedFiles/X_MAT_test_simpleIMP_constant_12h_on1line.csv'

train_data_reduced_path = 'ImputedFiles/train_data_iterImp_reduced.csv'
test_data_reduced_path = 'ImputedFiles/test_data_iterImp_reduced.csv'
train_data_reduced_withGrad_path = 'ImputedFiles/train_data_iterImp_reduced_withGrad.csv'
test_data_reduced_withGrad_path = 'ImputedFiles/test_data_iterImp_reduced_withGrad.csv'
train_data_imp_path = 'ImputedFiles/train_data_imp.csv'
test_data_imp_path = 'ImputedFiles/test_data_imp.csv'

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
#  If cond. to jump very timeconsuming simpleImpute step if Imputingfiles are provided.
if USE_SIMPLEIMP_FILES:
    print("Reading in simpleImpute files...")
    test_imp_constant_pd = pd.read_csv(TEST_SIMPLEIMP_CONST)
    train_imp_constant_pd = pd.read_csv(TRAIN_SIMPLEIMP_CONST)
    test_imp_mean_pd = pd.read_csv(TEST_SIMPLEIMP_MEAN)
    train_imp_mean_pd = pd.read_csv(TRAIN_SIMPLEIMP_MEAN)
    test_imp_median_pd = pd.read_csv(TEST_SIMPLEIMP_MEDIAN)
    train_imp_median_pd = pd.read_csv(TRAIN_SIMPLEIMP_MEDIAN)
    train_imp_constant_12h_on1line_pd = pd.read_csv(TRAIN_SIMPLEIMP_CONST_REDUCED)
    test_imp_constant_12h_on1line_pd = pd.read_csv(TEST_SIMPLEIMP_CONST_REDUCED)
else:
    print('Using simpleImpute. This can be very timeconsuming...')
    [test_imp_constant_pd, train_imp_constant_pd] = FeatureTransform_simpleImp.simpleimp_constant(test_features_pd, train_features_pd)
    [test_imp_mean_pd, train_imp_mean_pd] = FeatureTransform_simpleImp.simpleimp_mean(test_features_pd, train_features_pd)
    [test_imp_median_pd, train_imp_median_pd] = FeatureTransform_simpleImp.simpleimp_median(test_features_pd, train_features_pd)

# Get IterativeImpute data
if USE_ITERATIVEIMP_FILES:
    train_data_reduced_pd = pd.read_csv(train_data_reduced_path)
    test_data_reduced_pd = pd.read_csv(test_data_reduced_path)
    train_data_reduced_withGrad_pd = pd.read_csv(train_data_reduced_withGrad_path)
    test_data_reduced_withGrad_pd = pd.read_csv(test_data_reduced_withGrad_path)
    #train_data_imp_pd = pd.read_csv(train_data_imp_path) # Too large file for upload to Github. If you need this file, set USE_ITERATIVEIMP_FILES to False!
    test_data_imp_pd = pd.read_csv(test_data_imp_path)
else:
    train_data_reduced_pd, test_data_reduced_pd, train_data_imp_pd, test_data_imp_pd = FeatureTransform_IterativeImp.iterativeImpute(PATH_TRAIN_FEATURES, PATH_TEST_FEATURES, gradients_inactive)
    train_data_reduced_withGrad_pd, test_data_reduced_withGrad_pd, train_data_imp_pd, test_data_imp_pd = FeatureTransform_IterativeImp.iterativeImpute(PATH_TRAIN_FEATURES, PATH_TEST_FEATURES, gradients_active)



# -------------------------------------------------------------------------------------------------
# Run subtask1

print('=====   Imputing finished. Solving subtask1...')

# Extract Train Labels for Subtask 1
train_labels_task1_pd = train_labels_pd[['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']]

# Calculate Output for Subtask 1
result_subtask1_pd = Subtask1.solveSubtask1(train_data_reduced_pd, test_data_reduced_pd, train_labels_task1_pd)


# -------------------------------------------------------------------------------------------------
# Run subtask2

print('=====   Subtask1 finished. Solving subtask2...')

train_label_subtask2 = train_labels_pd['LABEL_Sepsis']
train_data_subtask2 = train_data_reduced_pd
del train_data_subtask2['pid'] # remove if reading in 12h_on1line file
del train_data_subtask2['Time']
test_data_subtask2 = test_data_reduced_pd
del test_data_subtask2['pid']
del test_data_subtask2['Time']

result_subtask2_pd = Subtask2.solveSubtask2(train_data_subtask2, train_label_subtask2, test_data_subtask2)


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
