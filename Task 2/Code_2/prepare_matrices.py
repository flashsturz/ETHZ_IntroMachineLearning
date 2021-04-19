#-----------------
# Some functions for Task2 IML2021

#--------------------------------------------------------------------------------------------------
#IMPORTS
#-------
import numpy as np
import pandas as pd

def prepare_Xmat(features_pd):
    list_pid = features_pd.pid.unique() #get list of all pid's in data
    X=np.empty((0,420))

    i=0
    for pid in list_pid:#[1,10,100,1000,10000,10002,10006,10007,1001,10010]: #
        append_X=features_pd.loc[features_pd['pid'] == pid].to_numpy()
        append_X=append_X[:,2:].flatten()
        X=np.vstack((X,append_X))

        #i=i+1
        #if(i%1000==0):
        #    print("Finished %d pid's for X matrix..." % i)
    X_np = np.array(X)

    print("Finished X-matrix.")

    return X_np

def prepare_Ymat(labels_pd):
    list_pid = labels_pd.pid.unique() #get list of all pid's in data
    y=np.empty((0,16))

    i=0
    for pid in list_pid:#[1,10,100,1000,10000,10002,10006,10007,1001,10010]: #
        append_y=labels_pd.loc[labels_pd['pid']==pid].to_numpy()
        y=np.vstack((y, append_y))

        #i=i+1
        #if(i%1000==0):
        #    print("Finished %d pid's for y matrix..." % i)
    y_np = y[:,12:16]

    print("Finished y-matrix.")

    return y_np

#--------------------------------------------------------------------------------------------------
#This function saves the X matrix to a csv for usage in the Estimator function.

#VARIABLES
FEATURE_MATRIX_FILE=[{
                      'name': 'mean_train',
                      'FILE_PATH':'train_features_simpleImpute_mean.csv',
                      'filename_output': 'X_MAT_train_features_simpleIMP_mean_12h_on1line.csv'},
                     {'name': 'median_train',
                      'FILE_PATH': 'train_features_simpleImpute_median.csv',
                      'filename_output': 'X_MAT_train_features_simpleIMP_median_12h_on1line.csv'},
                     {'name': 'constant_train',
                      'FILE_PATH': 'train_features_simpleImpute_constant.csv',
                      'filename_output': 'X_MAT_train_simpleIMP_constant_12h_on1line.csv'},
                     {'name': 'mean_test',
                      'FILE_PATH': 'test_features_simpleImpute_mean.csv',
                      'filename_output': 'X_MAT_test_simpleIMP_mean_12h_on1line.csv'},
                     {'name': 'median_test',
                      'FILE_PATH': 'test_features_simpleImpute_median.csv',
                      'filename_output': 'X_MAT_test_simpleIMP_median_12h_on1line.csv'},
                     {'name': 'constant_test',
                      'FILE_PATH': 'test_features_simpleImpute_constant.csv',
                      'filename_output': 'X_MAT_test_simpleIMP_constant_12h_on1line.csv'}] # set of csv-files contraining the features, in provided structure form problem statement

print('Execution starts...')
for feature_file in FEATURE_MATRIX_FILE:
    print('Starting with %s' % feature_file['name'])

    data_features = pd.read_csv(feature_file['FILE_PATH'])
    X_matrix=prepare_Xmat(data_features)

    pd.DataFrame(X_matrix).to_csv(feature_file['filename_output'],header=None,index=None)

