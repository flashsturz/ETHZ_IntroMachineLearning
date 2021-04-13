"""
###
input: train_features.csv
output: train_features_impute_knn.csv
###
"""

import numpy as np
from sklearn.impute import KNNImputer
import csv

# read in csv file, convert to list
filepath = '../Data_2/train_features'
with open(filepath + '.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader, None) # skips first row (col names)
    data = list(reader)


imputer = KNNImputer(n_neighbors=2)
knn_data = imputer.fit_transform(data)


np.savetxt(fname=filepath + "_imputed_knn_neig2_floatlen3.csv", X=knn_data, fmt='%1.3f', delimiter=",")


### to do: only do KNN Imputer for same Person (same id) (external loop)
