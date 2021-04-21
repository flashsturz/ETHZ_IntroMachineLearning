# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Description Task 2

#--------------------------------------------------------------------------------------------------

import numpy as np
import random
import FeatureTransform_simpleImp

[test_imp_constant_pd, train_imp_constant_pd] = FeatureTransform_simpleImp.simpleimp_constant('Data_2/test_features_SHORT_FOR_TESTING.csv','Data_2/train_features_SHORT_FOR_TESTING.csv')

test_imp_constant_pd.to_csv('testing_imp_constant.csv')

