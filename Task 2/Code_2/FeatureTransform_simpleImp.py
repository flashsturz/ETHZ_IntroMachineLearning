# Simon Tobler sitobler

# This Function implements methods to use for imputing the given patientdata in IML Task2 2021
# Use this file as importfile
# --------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import time as time


def md_list2pdSeries(inputlist, pd_col_list):
    [n_times, n_meas, n_pid] = np.shape(inputlist)
    stacked_list = inputlist[:, :, 0]
    for pid in range(1, n_pid):
        stacked_list = np.concatenate((stacked_list, inputlist[:, :, pid]), axis=0)

    pd_dataframe = pd.DataFrame(data=stacked_list, columns=pd_col_list)

    return pd_dataframe


def print_elapsed_time(starttime):
    time_now = time.perf_counter()
    elapsed_time = time_now-starttime
    print("Time elapsed since start: %.2f s" % elapsed_time)


def simple_imputer_iml2(strat, features_pd):

    # Writing full dataset to multidimensional np array:
    md_list = np.array([])

    fulldata = features_pd

    # From the pd_dataframe given by the csv, the 3d Array "md_list" is constructed.
    #    Dimensions: 0:time
    #                1:Labels
    #                2:pid
    list_pid = fulldata.pid.unique()  # get a list of all pids present in the dataset
    md_list = fulldata.loc[fulldata['pid'] == list_pid[0]].to_numpy()
    list_pid = np.delete(list_pid, 0)
    for pid in list_pid:
        this_data = fulldata.loc[fulldata['pid'] == pid].to_numpy()
        md_list = np.dstack((md_list, this_data))
    [n_times, n_meas, n_pid] = np.shape(md_list)

    # Other preparations
    # ------------------
    # Find averages over all patients for each measurement to get a bias value
    #   (needed to perform meadian or mean imputing in cols full of nan)
    fulldata_np = fulldata.to_numpy()
    (fulldata_rows, fulldata_cols) = np.shape(fulldata_np)
    avg_fulldata = np.nanmean(fulldata_np, axis=0)

    md_list_imp = md_list

    print("Finished prep. Imputing starts...")

    for pid in range(n_pid):
        this_pid = md_list[:, :, pid]

        if (strat != 'constant'):
            # Check if there is a col with all nan-values:
            isnan_bool = np.all(np.isnan(this_pid), axis=0)
            for isnan_col in range(np.shape(isnan_bool)[0]):
                if isnan_bool[isnan_col]:
                    # If all nan: isnan_col is true thus set one entry of col to avg over all patients:
                    this_pid[2, isnan_col] = avg_fulldata[isnan_col]

        # Imputers:
        imputer = SimpleImputer(missing_values=np.nan, strategy=strat)

        this_pid_imp = imputer.fit_transform(this_pid)

        # write imputed person data to the multi dimensional list

        md_list_imp[:, :, pid] = this_pid_imp

        # The md_list_imp needs to be written back to a pandas dataframe in the same shape as before.
        features_imp_pd = md_list2pdSeries(md_list_imp, fulldata.columns.tolist())

        return features_imp_pd


def simpleimp_mean(PATH_TEST_FEATURES, PATH_TRAIN_FEATURES):
    # This functions Imputes the test and train data using mean strategy and return pd_frames
    test_features_pd = pd.read_csv(PATH_TEST_FEATURES)
    train_features_pd = pd.read_csv(PATH_TRAIN_FEATURES)

    train_features_simpleimp = simple_imputer_iml2('mean', train_features_pd)
    test_features_simpleimp = simple_imputer_iml2('mean', test_features_pd)

    return train_features_simpleimp, test_features_simpleimp


def simpleimp_median(PATH_TEST_FEATURES, PATH_TRAIN_FEATURES):
    # This functions Imputes the test and train data using median strategy and return pd_frames

    test_features_pd = pd.read_csv(PATH_TEST_FEATURES)
    train_features_pd = pd.read_csv(PATH_TRAIN_FEATURES)

    train_features_simpleimp = simple_imputer_iml2('median', train_features_pd)
    test_features_simpleimp = simple_imputer_iml2('median', test_features_pd)

    return train_features_simpleimp, test_features_simpleimp


def simpleimp_constant(PATH_TEST_FEATURES, PATH_TRAIN_FEATURES):
    # This functions Imputes the test and train data using constant (0) strategy and return pd_frames
    test_features_pd = pd.read_csv(PATH_TEST_FEATURES)
    train_features_pd = pd.read_csv(PATH_TRAIN_FEATURES)

    train_features_simpleimp = simple_imputer_iml2('constant', train_features_pd)
    test_features_simpleimp = simple_imputer_iml2('constant', test_features_pd)

    return train_features_simpleimp, test_features_simpleimp

# --------------------------------------------------------------------------------------------------
# CODE
# ----

# NO CODE!!! USE AS IMPORTFILE.