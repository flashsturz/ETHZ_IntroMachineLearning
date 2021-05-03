# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#--------------------------------------------------------------------------------------------------
# Description Task 3
# Classification of mutations of a human antibody protein into active (1) and inactive (0) based on provided mutation information
# active mutation (1): protein retains original functions
# inactive mutation (0): protein looses its function
#--------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras


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

random.seed(1234)

current_time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
train_pd = pd.read_csv(PATH_TRAIN_DATA)
test_pd = pd.read_csv(PATH_TEST_DATA)

sample_pd = pd.read_csv(PATH_SAMPLE_FILE)



# ==================================================================================================
# VARIABLES
# ----------
