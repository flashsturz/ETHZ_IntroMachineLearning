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
import random
from datetime import datetime
import pandas as pd
import time
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score

# -------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------

def print_elapsed_time(starttime):
    time_now = time.perf_counter()
    elapsed_time = time_now-starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)

def add_splits(df):
    df_split = df['Sequence'].apply(lambda  x: pd.Series(list(x)))
    df_split.columns = ['S1', 'S2', 'S3', 'S4']
    concat = pd.concat([df_split, df], axis=1)
    concat.drop('Sequence', axis=1, inplace=True)
    return concat


# --------------------------------------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------------------------------------

PATH_TRAIN_DATA = '../Data_3/train.csv'
PATH_TEST_DATA = '../Data_3/test.csv'
PATH_SAMPLE_FILE = '../Data_3/sample.csv'


# --------------------------------------------------------------------------------------------------
# Reading Data, Preprocessing
# --------------------------------------------------------------------------------------------------
print('=====Execution starts.======')
print('=====   Preparations...')

random.seed(1234)

current_time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

print('=====   Reading data, preprocessing...')

# get pandas df
train_pd = pd.read_csv(PATH_TRAIN_DATA)
test_pd = pd.read_csv(PATH_TEST_DATA)
sample_pd = pd.read_csv(PATH_SAMPLE_FILE, header=None, names=['Active'])


# --------------------------------------------------------------------------------------------------
# get single characters rather than strings for column 'Sequence'
# --------------------------------------------------------------------------------------------------
train_split = add_splits(train_pd)
test_split = add_splits(test_pd)
print('train split head: \n', train_split.head())
print('test split head: \n', test_split.head())

# --------------------------------------------------------------------------------------------------
# convert features of train and test to numerical values
# --------------------------------------------------------------------------------------------------
print('=====   Encoding...')
enc = OrdinalEncoder()

train_ord = enc.fit_transform(train_split[['S1', 'S2', 'S3', 'S4']])
test_ord = enc.transform(test_split[['S1', 'S2', 'S3', 'S4']])

print(f'train ord ({type(train_ord)}, {train_ord.shape}): \n', train_ord)
print(f'test ord: ({type(test_ord)}, {test_ord.shape}): \n', test_ord)

# --------------------------------------------------------------------------------------------------
# get features and labels for further processing
# --------------------------------------------------------------------------------------------------
train_split_seq = pd.DataFrame(data=train_ord, columns=['S1', 'S2', 'S3', 'S4'])
test_split_seq = pd.DataFrame(data=test_ord, columns=['S1', 'S2', 'S3', 'S4'])

X_train = train_split_seq[['S1', 'S2', 'S3', 'S4']]
Y_train = train_split['Active'] # series
X_test = test_split_seq[['S1', 'S2', 'S3', 'S4']]

print('x shape: ',X_train.shape, type(X_train))
print('y shape: ',Y_train.shape, type(Y_train))
print('x test shape: ', X_test.shape, type(X_test))
