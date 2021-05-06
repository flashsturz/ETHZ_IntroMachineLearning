# Functions by flavio for Task3
# --------------------------------------------------------------------------------------------------

import numpy as np


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



# -------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------


def lettersToNumbers(train_data, test_data, train_labels, use_validation_set):
    # This function converts the Letters in the data to numbers in order to feed it to the Neural Network

    acids = np.array(
        [ord('A'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('K'), ord('L'), ord('M'),
         ord('N'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'), ord('U'), ord('V'), ord('W'), ord('Y')],
        dtype=object)
    categories_ = [acids, acids, acids,
                   acids]  # Assemble to array, where each column contains all the different letters, that the encoder should expect.
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

