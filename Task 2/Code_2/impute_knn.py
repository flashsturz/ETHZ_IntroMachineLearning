"""
###
input: train_features.csv
output: train_features_impute_knn.csv
###
"""

import numpy as np
from sklearn.impute import KNNImputer
import csv
import pandas as pd


def impute_jbaumer(path):
    # read in csv file, convert to list
    filepath = '../Data_2/test_features'
    filepath = path
    n_neighb = 2



    #with open(filepath + '.csv', newline='') as f:
    #    reader = csv.reader(f)
    #    next(reader, None) # skips first row (col names)
    #    data = list(reader)

    df_not_imputed = pd.read_csv(filepath+'.csv')
    df_imputed = pd.DataFrame(columns=cols)
    print(df_imputed.head())
    print(df_not_imputed.head())
    # loop over every person

    df_temp_not_imp = df_not_imputed[0:12]
    print(df_temp_not_imp)

    # nearest neighbor impute with KNN (for this person):
    imputer = KNNImputer(n_neighbors=n_neighb)
    # impute data per person and write back to pandas dataframe
    imputed_data_person = imputer.fit_transform(df_temp_not_imp)
    print("length imputed data person: ", len(imputed_data_person[0]))
    df_temp_imp = pd.DataFrame(imputed_data_person)

    # append to final df
    df_imputed = df_imputed.append(df_temp_imp)


    print("df temp first 12 not imputed: \n", df_temp_imp)
    print("df temp first 12 imputed: \n", df_imputed)

    """
    n_ppl = int(len(data)/12)
    print("number of ppl: ", n_ppl)

    ### to do: only do KNN Imputer for same Person (same id) (external loop)


    #print('type of knn_data: ', type(knn_data))


    # loop over list data[i] (consists of one row)
    # increase i by steps of 12 (which corresponds to a new person)
    # create a sublist of only one person
    # impute this sublist with KNN impute -> get a ndarray back
    # concatenate all ndarrays
    # save the concatenated ndarray to the file (e.g. as csv)

    
    imputer = KNNImputer(n_neighbors=n_neighb)
    knn_data = imputer.fit_transform(data)
    np.savetxt(fname=filepath + "_imputed_knn_neig" + str(n_neighb) + ".csv", X=knn_data, fmt='%1.3f', delimiter=",")
    """

    #return df
    return 0

# call fct
impute_jbaumer('../Data_2/test_features_small')