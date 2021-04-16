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
filepath = '../Data_2/train_features_3_ppl'
n_neighb = 2

with open(filepath + '.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader, None) # skips first row (col names)
    data = list(reader)


n_ppl = int(len(data)/12)
print("number of ppl: ", n_ppl)

### to do: only do KNN Imputer for same Person (same id) (external loop)
print('type of data: ', type(data))
print()
print()
print('data: ', data)
print()
print()
print('length of data: ', len(data))
print('length of data[0] (width of excel): ', len(data[0]))
print()
print()
print('data[0]: ', data[23])
print()

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

