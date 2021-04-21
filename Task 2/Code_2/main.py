# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Description Task 2

# -------------------------------------------------------------------------------------------------
# IMPORT
# ------
import numpy as np
import random
import FeatureTransform_simpleImp

# -------------------------------------------------------------------------------------------------
# Run imputer functions provided in imputer files.
#    Many pandas dataframes with imputed feature data.


[test_imp_constant_pd, train_imp_constant_pd] = FeatureTransform_simpleImp.simpleimp_constant('Data_2/test_features_SHORT_FOR_TESTING.csv','Data_2/train_features_SHORT_FOR_TESTING.csv')

test_imp_constant_pd.to_csv('testing_imp_constant.csv')  # only for testing


# -------------------------------------------------------------------------------------------------
# Run subtask functions provided in imputer files.

# subtask3_pd=subtask3(test_imp_constant_pd,train_imp_constant_pd) #TODO: create these functions in importfiles.