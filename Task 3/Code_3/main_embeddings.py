# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#--------------------------------------------------------------------------------------------------
# Description Task 3
# Classification of mutations of a human antibody protein into active (1) and inactive (0) based on provided mutation information
# active mutation (1): protein retains original functions
# inactive mutation (0): protein looses its function
#--------------------------------------------------------------------------------------------------

import numpy as np
import time
import random

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.metrics import Recall, Precision

# -------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------

def print_elapsed_time(starttime):
    time_now = time.perf_counter()
    elapsed_time = time_now-starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)
    
def lettersToNumbers(train_data, test_data, train_labels, use_validation_set):
    # This function converts the Letters in the data to numbers in order to feed it to the Neural Network
    acids = np.array([ord('A'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('K'), ord('L'), ord('M'), ord('N'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'), ord('U'), ord('V'), ord('W'), ord('Y')], dtype = object)
    categories_ = [acids, acids, acids, acids] # Assemble to array, where each column contains all the different letters, that the encoder should expect.
    numbers_train = []
    numbers_test = []
    
    if use_validation_set == True: # We split up the training data into a train and a validation set
        # Split the training data into train and validation set, if activated
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.2, random_state = 42)
        numbers_val = []
        for row in val_data:
            convertedMutationNumber_val = [ord(row[0]), ord(row[1]), ord(row[2]), ord(row[3])] # Convert Characters to their unicode number (A is unicode 65)
            numbers_val.append(convertedMutationNumber_val)
    
    for row in train_data:
        # Split the 4 character sequence into the single characters for each row and store their unicode number
        convertedMutationNumber_train = [ord(row[0]), ord(row[1]), ord(row[2]), ord(row[3])] # Convert Characters to their unicode number (A is unicode 65)
        numbers_train.append(convertedMutationNumber_train)
    for row in test_data:
        convertedMutationNumber_test = [ord(row[0]), ord(row[1]), ord(row[2]), ord(row[3])] # Convert Characters to their unicode number (A is unicode 65)
        numbers_test.append(convertedMutationNumber_test)
    
    # Encode the values using the onehotencoder
    #enc = OneHotEncoder(handle_unknown = 'ignore') # Instead of Ignore we might directly pass the categories (if we encounter an unseen label in the test set, 'ignore' just sets this label to zero)
    #enc = OneHotEncoder(categories = categories_)
    enc = OneHotEncoder()
    
    enc.fit(numbers_train) # Fit only on Training Data, best practice 
    converted_train_Data = enc.transform(numbers_train).toarray()
    converted_test_Data = enc.transform(numbers_test).toarray()
    if use_validation_set == True: # only necessary if we use validation data
        converted_val_Data = enc.transform(numbers_val).toarray()
        return converted_train_Data, converted_test_Data, converted_val_Data, train_labels, val_labels
    else:
        return converted_train_Data, converted_test_Data, train_labels

# -------------------------------------------------------------------------------------------------
# VARIABLES
# ---------

PATH_TRAIN_DATA = '../Data_3/train.csv'
PATH_TEST_DATA = '../Data_3/test.csv'

use_validation_set = True # Toggle for Validation Set creation
use_weighted_fitting = False # Use Class Weights for Fitting --> https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

# -------------------------------------------------------------------------------------------------
# PREPS
# -----
starttime = time.perf_counter()

print('=====Execution starts.======')
print('=====   Reading in Data...')
print_elapsed_time(starttime)

random.seed(1234)

train_data_pd = pd.read_csv(PATH_TRAIN_DATA)
neg, pos = np.bincount(train_data_pd['Active']) # Number of negative and positive train datas
total = neg + pos
# Compute Class Weights for fitting
weight_for_0 = (1/neg) * (total) / 2.0
weight_for_1 = (1/pos) * total / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}

test_data_pd = pd.read_csv(PATH_TEST_DATA)

train_labels_pd = train_data_pd['Active']
train_features_pd = train_data_pd['Sequence']

test_features_pd = test_data_pd['Sequence']


print('=====   Encoding features...')
print_elapsed_time(starttime)
if use_validation_set == True:
    [encoded_train_features, encoded_test_features, encoded_val_features, train_labels, val_labels] = lettersToNumbers(train_features_pd.values, test_features_pd.values, train_labels_pd.values, use_validation_set)
else:
    [encoded_train_features, encoded_test_features, train_labels] = lettersToNumbers(train_features_pd.values, test_features_pd.values, train_labels_pd.values, use_validation_set)

print('=====   Prepare Model...')
print_elapsed_time(starttime)
model = Sequential()
model.add(Embedding(input_dim = 21, output_dim = 40, input_length = encoded_train_features.shape[1])) # Add embedding layer that has full vocabulary of all characters
model.add(Flatten())
model.add(Dense(150, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dropout(rate= 0.2, seed = 42 )) # Droput layers prevent overfitting
model.add(Dense(300, activation = 'relu', kernel_initializer = 'he_normal')) # Relu Activation is better than sigmoid, according to machinelearningmastery
model.add(Dropout(rate= 0.2, seed = 42 ))
model.add(Dense(150, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dropout(rate= 0.2, seed = 42 ))
model.add(Dense(1, activation = 'sigmoid')) # Sigmoid output, we need to convert this to a binary output with the round() function

print('=====   Compile Model...')
print_elapsed_time(starttime)
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = [Precision(), Recall()])

print(model.summary())

print('=====   Fit Model...')
print_elapsed_time(starttime)
if use_validation_set == True:
    if use_weighted_fitting == True:
        model.fit(encoded_train_features, train_labels, epochs = 60, batch_size = 32, validation_data = (encoded_val_features, val_labels), class_weight = class_weight, verbose = 2, workers = 4)
    else:
        model.fit(encoded_train_features, train_labels, epochs = 30, batch_size = 32, validation_data = (encoded_val_features, val_labels), verbose = 2, workers = 4)
        # Batch size: https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/
else:
    if use_weighted_fitting == True:
        model.fit(encoded_train_features, train_labels, epochs = 20, batch_size = 64, validation_split = 0.2, class_weight = class_weight, verbose = 2, workers = 4)
    else:
        model.fit(encoded_train_features, train_labels, epochs = 20, batch_size = 64, validation_split = 0.2, verbose = 2, workers = 4)

print('=====   Evaluate Model...')
print_elapsed_time(starttime)
if use_validation_set == True:
    # Calculate the f1_score on the validation data set
    loss, precision, recall = model.evaluate(encoded_val_features, val_labels, verbose = 0)
    manual_f1_score = 2*((precision*recall)/(precision+recall+K.epsilon())) # calculate f1_score manually with precision and recall
    y_pred_val = np.round(model.predict(encoded_val_features, workers = 2))

    print(classification_report(val_labels, y_pred_val)) # print report
    F1_score_sklearn = f1_score(val_labels, y_pred_val) # calculate f1_score through SK learn
    print(f'F1 Score on Validation: {F1_score_sklearn}')
    print_elapsed_time(starttime)

# Write prediction of test data to output file
y_pred = np.round(model.predict(encoded_test_features, workers = 2))
np.savetxt('prediction_neural_embeddings.csv', y_pred, delimiter='\n', fmt ='%d')