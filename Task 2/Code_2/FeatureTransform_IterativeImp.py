# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
# -----------------
# Imputer Function File for Iterative Impute
# Use this only as importfile
# --------------------------------------------------------------------------------------------------


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import pandas as pd
import numpy as np
import os
from scipy import stats
import time


def combineHourlyMeasurementsPerPatient(train_data, test_data):
    # Reduce the data from the 12 hourly measurements per patient to one row per patient using the mean of the 12 entries per feature
    # Assumes imputed input data without missing values
    train_data_comb, test_data_comb = [], []
    patients = int(train_data.shape[0] / 12)
    print(f'Patients: {patients}')
    # for training data
    for i in range(0, patients):
        data_patient = train_data[12 * (i):12 * (i + 1), :]
        data_mean = data_patient.mean(axis=0)
        train_data_comb.append(data_mean)
    # for test data
    for i in range(0, int(test_data.shape[0] / 12)):
        data_patient = test_data[12 * (i):12 * (i + 1), :]
        data_mean = data_patient.mean(axis=0)
        test_data_comb.append(data_mean)
    return np.asarray(train_data_comb), np.asarray(test_data_comb)


def FeatureGradients(data_set, data_type):
    # Calculate the rate of change of every feature for every patient as an additional feature
    # Assumes imputed input data without missing values
    gradients = []

    #TODO: Flavio: Zeile 44-49 durch 51 ersetzen korrekt? (train_data nicht vorhanden.)
    #patients = int(train_data.shape[0] / 12)

    #if data_type == 'train':
    #    N = patients
    #if data_type == 'test':
    #    N = int(data_set.shape[0] / 12)

    N = int(data_set.shape[0] / 12)

    for i in range(0, N):
        # data_patient = train_data[12 * (i):12 * (i + 1), :] #TODO:Flavio: Zeile 54 durch Zeile55 ersetzt, korrekt? (Train_data nicht gegeben.)
        data_patient = data_set[12 * (i):12 * (i + 1), :]
        patient_gradient = [0] * (data_set.shape[1] - 3)  # Since we omit the first three columns in the gradient calculation

        for j in range(3, data_set.shape[1]):
            y_vec = data_patient[:, j]
            x_vec = range(0, len(y_vec))
            if y_vec.any():
                if len(y_vec) > 1:
                    slope, intercept, r_value, p_value, std_er = stats.linregress(x_vec, y_vec)
                    if np.isnan(slope):
                        slope = 0
                else:
                    slope = 0
            else:
                slope = 0
            patient_gradient[j - 3] = slope
        gradients.append(patient_gradient)
    return gradients


def print_elapsed_time(starttime):  # Print Elapsed Time
    time_now = time.perf_counter()
    elapsed_time = time_now - starttime
    print("Time elapsed since start: %.2f s" % elapsed_time)


###############################################################################

def iterativeImpute(train_data_path, test_data_path, use_gradients):
    # toggle data import
    data_import_active = True

    print(f'Starts. Data Import = {data_import_active}, Use Gradients = {use_gradients}')
    totaltime_start = time.perf_counter()

    # Read in Data and Processing
    if data_import_active or not (os.path.isfile('../Data_2/train_features_iterImp.csv') and os.path.isfile(
            '../Data_2/train_features_reduced.csv')):
        train_data_frame = pd.read_csv(train_data_path)
        test_data_frame = pd.read_csv(test_data_path)

        train_data = train_data_frame.values
        test_data = test_data_frame.values

        # Iterative Impute on training data
        print("Finished Preparation")
        print_elapsed_time(totaltime_start)

        #        if not os.path.isfile('../Data_2/train_features_iterImp.csv'):
        print("Since no imputed Data found, calculate Imputed Dataset...")
        imputer = IterativeImputer(random_state=42)
        train_data_imp = imputer.fit_transform(train_data)
        test_data_imp = imputer.transform(test_data)  # Transform test_data with same rules as train data

        #np.savetxt('../Data_2/train_features_iterImp.csv', train_data_imp, fmt=('%.3f'), delimiter=',', comments='', )  #TODO Flavio: Habe das kommentiert, gab fehler "No such file or directory: ..."
        #np.savetxt('../Data_2/test_features_iterImp.csv', test_data_imp, fmt=('%.3f'), delimiter=',', comments='', )  #TODO Flavio: Habe das kommentiert, gab fehler "No such file or directory: ..."
        #        if os.path.isfile('../Data_2/train_features_iterImp.csv'):
        #            print("Imputed Data Set found, read in File")
        #            train_data_imp = np.genfromtxt('../Data_2/train_features_imp.csv', dtype=float, delimiter=',')
        #            test_data_imp = np.genfromtxt('../Data_2/test_features_imp.csv', dtype=float, delimiter=',')

        train_data_imp_pd = pd.DataFrame(train_data_imp, columns=train_data_frame.columns)
        test_data_imp_pd = pd.DataFrame(test_data_imp, columns=train_data_frame.columns)

        train_data_imp_pd.to_csv('ImputedFiles/train_data_imp.csv')
        test_data_imp_pd.to_csv('ImputedFiles/test_data_imp.csv')

        print("Finished Imputation. Reducing Data Starts")
        print_elapsed_time(totaltime_start)

        #        if not os.path.isfile('../Data_2/train_features_iterImp_reduced.csv'):
        print("No Reduced Data Set found, compute reduced set...")
        train_data_reduced, test_data_reduced = combineHourlyMeasurementsPerPatient(train_data_imp, test_data_imp)
        #np.savetxt('../Data_2/train_features_iterImp_reduced.csv', train_data_reduced, fmt=('%.3f'), delimiter=',',
        #           comments='', )  #TODO Flavio: Habe das kommentiert, gab fehler "No such file or directory: ..."
        #np.savetxt('../Data_2/test_features_iterImp_reduced.csv', test_data_reduced, fmt=('%.3f'), delimiter=',',
        #           comments='', )  #TODO Flavio: Habe das kommentiert, gab fehler "No such file or directory: ..."
        #        else:
        #            print("Reduced Data Set found, read in Data...")
        #            train_data_reduced = np.genfromtxt('../Data_2/train_features_iterImp_reduced.csv', dtype=float, delimiter=',')
        #            test_data_reduced = np.genfromtxt('../Data_2/test_features_iterImp_reduced.csv', dtype=float, delimiter=',')
        print("Finished Reduction of Data.")
        print_elapsed_time(totaltime_start)

        train_data_reduced_pd = pd.DataFrame(train_data_reduced, columns=train_data_frame.columns)
        test_data_reduced_pd = pd.DataFrame(test_data_reduced, columns=test_data_frame.columns)

        train_data_reduced_pd.to_csv('ImputedFiles/train_data_iterImp_reduced.csv')
        test_data_reduced_pd.to_csv('ImputedFiles/test_data_iterImp_reduced.csv')

        #return train_data_reduced_pd, test_data_reduced_pd, train_data_imp_pd, test_data_imp_pd #TODO: Flavio: Muss diese Zeile nicht ans ende (Z169)? Wenn hier werden die folgenden if nicht ausgeführt.

    if use_gradients:
        print("Gradients:")
        #        if not os.path.isfile('../Data_2/train_features_iterImp_reduced_withGrad.csv'):
        print("No Gradient Set found, calculating Gradients...")
        train_gradients = FeatureGradients(train_data_imp, 'train')
        test_gradients = FeatureGradients(test_data_imp, 'test')
        train_data_reduced_withGrad = np.c_[train_data_reduced, train_gradients]
        test_data_reduced_withGrad = np.c_[test_data_reduced, test_gradients]
        # np.savetxt('../Data_2/train_features_iterImp_reduced_withGrad.csv', train_data_reduced_withGrad, fmt=('%.3f'),
        #            delimiter=',', comments='', )  #TODO Flavio: Habe das kommentiert, gab fehler "No such file or directory: ..."
        # np.savetxt('../Data_2/test_features_iterImp_reduced_withGrad.csv', test_data_reduced_withGrad, fmt=('%.3f'),
        #            delimiter=',', comments='', )  #TODO Flavio: Habe das kommentiert, gab fehler "No such file or directory: ..."
        #        else:
        #            print("Gradient Data Set found, read in Data...")
        #            train_data_reduced_withGrad = np.genfromtxt('../Data_2/train_features_iterImp_reduced_withGrad.csv', dtype=float, delimiter=',')
        #            test_data_reduced_withGrad = np.genfromtxt('../Data_2/test_features_iterImp_reduced_withGrad.csv', dtype=float, delimiter=',')
        print("Finished Gradients. Finished Imputation.")
        print_elapsed_time(totaltime_start)

        gradient_labels = ['Grad_' + s for s in train_data_frame.columns[3:]]

        train_data_reduced_withGrad_pd = pd.DataFrame(train_data_reduced_withGrad,
                                                      columns=[*train_data_frame.columns, *gradient_labels])
        test_data_reduced_withGrad_pd = pd.DataFrame(test_data_reduced_withGrad,
                                                     columns=[*test_data_frame.columns, *gradient_labels])

        train_data_reduced_withGrad_pd.to_csv('ImputedFiles/train_data_iterImp_reduced_withGrad.csv')
        test_data_reduced_withGrad_pd.to_csv('ImputedFiles/test_data_iterImp_reduced_withGrad.csv')

        return train_data_reduced_withGrad_pd, test_data_reduced_withGrad_pd, train_data_imp_pd, test_data_imp_pd

    return train_data_reduced_pd, test_data_reduced_pd, train_data_imp_pd, test_data_imp_pd  #TODO Flavio: Hier richtig??

###############################################################################
