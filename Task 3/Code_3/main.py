# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
# --------------------------------------------------------------------------------------------------
# Description Task 3
# Classification of mutations of a human antibody protein into active (1) and inactive (0) based on provided
# mutation information
# active mutation (1): protein retains original functions
# inactive mutation (0): protein looses its function
# --------------------------------------------------------------------------------------------------


import random
from datetime import datetime
import time as time
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Recall, Precision

import Simon

# -------------------------------------------------------------------------------------------------
# VARIABLES
# ---------

PATH_TRAIN_DATA = 'Data_3/train.csv'
PATH_TEST_DATA = 'Data_3/test.csv'
PATH_SAMPLE_FILE = 'Data_3/sample.csv'

current_time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


# -------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------
def print_elapsed_time(starttime):
    time_now = time.perf_counter()
    elapsed_time = time_now - starttime
    print("    Time elapsed since start: %.2f s" % elapsed_time)


def lettersToNumbers(train_data, test_data, train_labels, use_validation_set):
    # This function converts the Letters in the data to numbers in order to feed it to the Neural Network

    acids = np.array(
        [ord('A'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('K'), ord('L'), ord('M'),
         ord('N'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'), ord('U'), ord('V'), ord('W'), ord('Y')],
        dtype=object)
    categories_ = [acids, acids, acids,
                   acids]  # Assemble to array, each col contains all the different letters, that the encoder expects.
    numbers_train = []
    numbers_test = []

    if use_validation_set == True:  # We split up the training data into a train and a validation set
        # Split the training data into train and validation set, if activated
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                          random_state=42)
        numbers_val = []
        for row in val_data:
            convertedMutationNumber_val = [ord(row[0]), ord(row[1]), ord(row[2]),
                                           ord(row[3])]  # Convert Characters to their unicode number (A is unicode 65)
            numbers_val.append(convertedMutationNumber_val)

    for row in train_data:
        # Split the 4 character sequence into the single characters for each row and store their unicode number
        convertedMutationNumber_train = [ord(row[0]), ord(row[1]), ord(row[2]),
                                         ord(row[3])]  # Convert Characters to their unicode number (A is unicode 65)
        numbers_train.append(convertedMutationNumber_train)
    for row in test_data:
        convertedMutationNumber_test = [ord(row[0]), ord(row[1]), ord(row[2]),
                                        ord(row[3])]  # Convert Characters to their unicode number (A is unicode 65)
        numbers_test.append(convertedMutationNumber_test)

    # Encode the values using the onehotencoder
    # enc = OneHotEncoder(handle_unknown = 'ignore') # Instead of Ignore we might directly pass the categories (if we encounter an unseen label in the test set, 'ignore' just sets this label to zero)
    # enc = OneHotEncoder(categories = categories_)
    enc = OneHotEncoder()

    enc.fit(numbers_train)  # Fit only on Training Data, best practice
    converted_train_Data = enc.transform(numbers_train).toarray()
    converted_test_Data = enc.transform(numbers_test).toarray()
    if use_validation_set == True:  # only necessary if we use validation data
        converted_val_Data = enc.transform(numbers_val).toarray()
        return converted_train_Data, converted_test_Data, converted_val_Data, train_labels, val_labels
    else:
        return converted_train_Data, converted_test_Data, train_labels

def model_flavio():
    print('    Creating model - model flavio...')
    print_elapsed_time(time_start)

    model = Sequential()
    model.add(Dense(150, input_dim=80, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(rate=0.2, seed=42))  # Droput layers prevent overfitting
    model.add(Dense(300, activation='relu',
                    kernel_initializer='he_normal'))  # Relu is better than sigmoid, according to machinelearningmastery
    model.add(Dropout(rate=0.2, seed=42))
    model.add(Dense(150, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(rate=0.2, seed=42))
    model.add(Dense(1,
                    activation='sigmoid'))  # Sigmoid output, we need to convert this to a binary output with round()

    print('=====   Compile Model...')
    print_elapsed_time(time_start)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Precision(), Recall()])

    return model


def model_wodropout():
    print('    Creating model - model without dropout...')
    print_elapsed_time(time_start)

    model = Sequential()
    model.add(Dense(150, input_dim=80, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(300, activation='relu',
                    kernel_initializer='he_normal'))  # Relu is better than sigmoid, according to machinelearningmastery
    model.add(Dense(150, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1,
                    activation='sigmoid'))  # Sigmoid output, we need to convert this to a binary output with round()

    print('=====   Compile Model...')
    print_elapsed_time(time_start)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Precision(), Recall()])

    return model


def model_flex_simon(input_dim, layer_list, model_name):
    inputs = keras.Input(shape=input_dim)
    x = keras.layers.Dense(10, activation='relu')(inputs)

    for entry in layer_list:
        x = keras.layers.Dense(entry['n_units'], activation=entry['activation'], name=entry['name'])(x)
        if entry['addDropout']:
            x = keras.layers.Dropout(rate=0.2)(x)

    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.summary()

    print_elapsed_time(time_start)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Precision(), Recall()])

    return model

def fit_evaluate_model(model, x_train, y_train, x_val, y_val, modelname):
    model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val),
              verbose=2, workers=4)

    # Calculate the f1_score on the validation data set
    loss, precision, recall = model.evaluate(x_val, y_val, verbose=0)
    manual_f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))  # calculate f1_score manually with precision and recall
    y_pred_val = np.round(model.predict(x_val, workers=2))

    report = classification_report(y_val, y_pred_val)
    F1_score_sklearn = f1_score(y_val, y_pred_val)  # calculate f1_score through SK learn
    print(f'F1 Score on Validation: {F1_score_sklearn}')

    y_pred = np.round(model.predict(x_test_flavio, workers=2))

    this_results = {'ModelName': [modelname], 'F1_score': [F1_score_sklearn]}
    this_results = pd.DataFrame.from_dict(this_results)

    return this_results, y_pred


# ==================================================================================================
# DATA STUFF
# ----------

time_start = time.perf_counter()

report = pd.DataFrame(data=None, columns=['ModelName', 'F1_score', 'Elapsed Time [s]'])

print('=====Execution starts.======')
print('=====   Preparations...')
current_time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

random.seed(1234)

train_pd = pd.read_csv(PATH_TRAIN_DATA)
test_pd = pd.read_csv(PATH_TEST_DATA)

sample_pd = pd.read_csv(PATH_SAMPLE_FILE)

[seq, fullseq, act] = Simon.bdpsimon_train2letters(train_pd)
[x_train_flavio, x_test_flavio, x_val_flavio, y_train_flavio, y_val_flavio] = lettersToNumbers(
    train_pd['Sequence'].to_numpy(),
    test_pd['Sequence'].to_numpy(),
    train_pd['Active'].to_numpy(), True)

print("Shape of seq is: ", np.shape(seq))
print("Shape of act is: ", np.shape(act))
print("Shape of x_train_flavio is: ", np.shape(x_train_flavio))
print("Shape of x_test_flavio is: ", np.shape(x_test_flavio))
print("Shape of x_val_flavio is: ", np.shape(x_val_flavio))
print("Shape of y_train_flavio is: ", np.shape(y_train_flavio))
print("Shape of y_val_flavio is: ", np.shape(y_val_flavio))


print('=====   Model preparation...')
print_elapsed_time(time_start)

layer_list_1 = [{'name': 'dense1', 'n_units': 50, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense2', 'n_units': 50, 'activation': 'relu', 'addDropout': True}]
layer_list_2 = [{'name': 'dense1', 'n_units': 100, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense2', 'n_units': 200, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense3', 'n_units': 100, 'activation': 'relu', 'addDropout': True}]
layer_list_3 = [{'name': 'dense1', 'n_units': 10, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense21', 'n_units': 20, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense22', 'n_units': 20, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense23', 'n_units': 20, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense24', 'n_units': 20, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense25', 'n_units': 20, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense3', 'n_units': 10, 'activation': 'relu', 'addDropout': True}]
layer_list_large = [{'name': 'dense1', 'n_units': 10, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense2', 'n_units': 100, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense6', 'n_units': 1000, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense3', 'n_units': 100, 'activation': 'relu', 'addDropout': True},
                {'name': 'dense4', 'n_units': 10, 'activation': 'relu', 'addDropout': True}]

model_list=[{'modelname': 'modelsmall', 'layerlist': layer_list_1},
            {'modelname': 'modelstd', 'layerlist': layer_list_2},
            {'modelname': 'modeldeep', 'layerlist': layer_list_3},
            {'modelname': 'modelpyramid', 'layerlist': layer_list_large}]

for model in model_list:
    modelname=model['modelname']
    layerlist=model['layerlist']

    print('==== Starting with Model ', modelname)
    print_elapsed_time(time_start)

    model = model_flex_simon(80,layer_list=layerlist, model_name=modelname)
    this_results, y_pred = fit_evaluate_model(model, x_train_flavio, y_train_flavio, x_val_flavio, y_val_flavio,modelname=modelname)

    np.savetxt('prediction_neural'+modelname+ '_' +current_time_str+'_.csv', y_pred, delimiter='\n', fmt='%d')

    time_now = time.perf_counter()
    elapsed_time = time_now - time_start

    this_results['Elapsed Time [s]']=elapsed_time

    report = report.append(this_results, ignore_index=True)
    print('==== Finished... ', modelname)
    print_elapsed_time(time_start)


print('==== Finished all, exiting.')
report.to_csv('results' + current_time_str + '.csv')