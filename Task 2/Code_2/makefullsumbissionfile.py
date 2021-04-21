# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
# -----------------
# Task 2
# This function constructs a full submission file out of the 3 prediction files fro the subtasks.

# Add the Files with the predictions as arguments to the functions in the folowing manner_
#   Arg1: Prediction of subtask1
#   Arg2: Prediction of subtask2
#   Arg3: Prediction of subtask3
# ATTENTION: The files need to be in the same format as sample.csv, the predictions at the correct positions in the file
#            All the other values can be anything.

# --------------------------------------------------------------------------------------------------
# IMPORTS
# -------


import numpy as np
import pandas as pd
import time as time
from datetime import datetime
import sys

# --------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------


def print_elapsed_time(starttime):
    time_now = time.perf_counter()
    elapsed_time = time_now-starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)


def print_file_error_and_exit():
    print('ERROR!')
    print("Something is wrong with your provided files, some are missing!")
    print("Usage of this func: $python3 makefullsubmissionfile.py "
          "<path-to-prediction-of-subtask1> <path-to-prediction-of-subtask2> <path-to-prediction-of-subtask3>")
    exit()


# --------------------------------------------------------------------------------------------------
# VARIABLES
# ---------


COL_SUBTASK1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
                'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
COL_SUBTASK2 = ['LABEL_Sepsis']
COL_SUBTASK3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

COL_ALL = ['pid']+COL_SUBTASK1+COL_SUBTASK2+COL_SUBTASK3

FILE_SUBTASK1 = sys.argv[1] if len(sys.argv) >= 2 else print_file_error_and_exit()
FILE_SUBTASK2 = sys.argv[2] if len(sys.argv) >= 3 else print_file_error_and_exit()
FILE_SUBTASK3 = sys.argv[3] if len(sys.argv) >= 4 else print_file_error_and_exit()

SAMPLE_FILE='Data_2/sample.csv'

TEST_FEATURES = 'Data_2/test_features.csv'

current_time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

# --------------------------------------------------------------------------------------------------
# CODE
# ---------

subtask1_pd = pd.read_csv(FILE_SUBTASK1)
subtask2_pd = pd.read_csv(FILE_SUBTASK2)
subtask3_pd = pd.read_csv(FILE_SUBTASK3)
sample_pd=pd.read_csv(SAMPLE_FILE)

test_features_pd = pd.read_csv(TEST_FEATURES)
list_pid = test_features_pd.pid.unique()  # get list of all pid in test_data

full_submission_pd = pd.DataFrame(data=np.zeros((len(list_pid),len(sample_pd.columns))), columns=sample_pd.columns)

full_submission_pd['pid'] = np.array(list_pid).T

benchmark_st1_pd=full_submission_pd
benchmark_st2_pd=full_submission_pd
benchmark_st3_pd=full_submission_pd
# for col in COL_SUBTASK1:
#    print(subtask1_pd[col])
full_submission_pd[COL_SUBTASK1] = subtask1_pd[COL_SUBTASK1]
full_submission_pd[COL_SUBTASK2] = subtask2_pd[COL_SUBTASK2]
full_submission_pd[COL_SUBTASK3] = subtask3_pd[COL_SUBTASK3]
benchmark_st1_pd[COL_SUBTASK1] = subtask1_pd[COL_SUBTASK1]
benchmark_st2_pd[COL_SUBTASK2] = subtask2_pd[COL_SUBTASK2]
benchmark_st3_pd[COL_SUBTASK3] = subtask3_pd[COL_SUBTASK3]

pd.DataFrame(full_submission_pd).to_csv("Submissionfiles/full_submission" + current_time_str + ".csv",index=False,float_format='%.3f')
pd.DataFrame(full_submission_pd).to_csv("full_submission" + current_time_str + ".zip", index=False,
                                        float_format='%.3f',
                                        compression='zip')
pd.DataFrame(benchmark_st1_pd).to_csv("benchmark_1" + current_time_str + ".zip", index=False,
                                        float_format='%.3f',
                                        compression='zip')
pd.DataFrame(benchmark_st1_pd).to_csv("benchmark_1" + current_time_str + ".csv", index=False,
                                        float_format='%.3f')
pd.DataFrame(benchmark_st2_pd).to_csv("benchmark_2" + current_time_str + ".zip", index=False,
                                        float_format='%.3f',
                                        compression='zip')
pd.DataFrame(benchmark_st2_pd).to_csv("benchmark_2" + current_time_str + ".csv", index=False,
                                        float_format='%.3f')
pd.DataFrame(benchmark_st3_pd).to_csv("benchmark_3" + current_time_str + ".zip", index=False,
                                        float_format='%.3f',
                                        compression='zip')
pd.DataFrame(benchmark_st3_pd).to_csv("benchmark_3" + current_time_str + ".csv", index=False,
                                        float_format='%.3f')
