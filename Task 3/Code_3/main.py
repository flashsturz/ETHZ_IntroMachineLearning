# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#--------------------------------------------------------------------------------------------------
# Description Task 3
# Classification of mutations of a human antibody protein into active (1) and inactive (0) based on provided mutation information
# active mutation (1): protein retains original functions
# inactive mutation (0): protein looses its function
#--------------------------------------------------------------------------------------------------

import numpy as np
import random
from datetime import datetime
import time as time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras

import Simon
import Flavio


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

PATH_TRAIN_DATA = 'Data_3/train.csv'
PATH_TEST_DATA = 'Data_3/test.csv'
PATH_SAMPLE_FILE = 'Data_3/sample.csv'
# ==================================================================================================
# DATA STUFF
# ----------
print('=====Execution starts.======')
print('=====   Preparations...')
current_time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

random.seed(1234)

train_pd = pd.read_csv(PATH_TRAIN_DATA)
test_pd = pd.read_csv(PATH_TEST_DATA)

sample_pd = pd.read_csv(PATH_SAMPLE_FILE)

[seq, fullseq, act] = Simon.bdpsimon_train2letters(train_pd)
[x_train_flavio, x_test_flavio, x_val_flavio, y_train_flavio, y_val_flavio] = Flavio.lettersToNumbers(train_pd['Sequence'].to_numpy(),
                                                                                                      test_pd['Sequence'].to_numpy(),
                                                                                                      train_pd['Active'].to_numpy(), True)

print("Shape of seq is: ", np.shape(seq))
print("Shape of act is: ", np.shape(act))
print("Shape of x_train_flavio is: ", np.shape(x_train_flavio))
print("Shape of x_test_flavio is: ", np.shape(x_test_flavio))
print("Shape of x_val_flavio is: ", np.shape(x_val_flavio))
print("Shape of y_train_flavio is: ", np.shape(y_train_flavio))
print("Shape of y_val_flavio is: ", np.shape(y_val_flavio))

simpleModel=Simon.keras_getmodel(x_train_flavio,y_train_flavio)
#test = Simon.keras_test(4, fullseq, act)
