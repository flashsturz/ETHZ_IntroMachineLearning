# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Task 1b: Perform linear regression on given data-set with given feature functions
# https://scikit-learn.org/stable/modules/linear_model.html

#--------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import RepeatedKFold
from math import sqrt

class data:
    def __init__(self,file,ycol,xcol_start):
        self.dataset=pd.read_csv(file)
        self.np_data=self.dataset.to_numpy()
        #self.X=np.delete(self.np_data,ycol,axis=1)
        self.X=self.np_data[:,xcol_start:]
        self.y=self.np_data[:,ycol]

def prepare_data(linear_X):
    squared_X=np.square(linear_X)
    exp_X=np.exp(linear_X)
    cos_X=np.cos(linear_X)
    const_X=np.ones((linear_X.shape[0],1))
    Phi=np.concatenate((linear_X,squared_X,exp_X,cos_X,const_X),axis=1)
    return Phi

#----------------------------------------------------------------------------------------------------------------------
# Read in Data
train_data=data('../Data_1b/train.csv',1,2)
x_train = train_data.X
y_train = train_data.y

Phi=prepare_data(train_data.X) # Regressor Matrix unscaled

# Analyse the range of values of the different x_i
variability_x = np.matrix([[max(x_train[:, 0]), max(x_train[:, 1]), max(x_train[:, 2]), max(x_train[:, 3]), max(x_train[:, 4])], [min(x_train[:, 0]), min(x_train[:, 1]), min(x_train[:, 2]), min(x_train[:, 3]), min(x_train[:, 4])]])

# Testing with scaling https://scikit-learn.org/stable/modules/preprocessing.html 
#------------------------------------------------------------------------------
# Version1: Scale Regressor Matrix only
scaler_phi = preprocessing.StandardScaler().fit(Phi) # Initiate Standardization
Phi_scaled = scaler_phi.transform(Phi)

# Version 2: Scale X and Y raw data
scaler_x = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler_x.transform(x_train)

Phi_x_scaled = prepare_data(x_train_scaled)
#scaler_y = preprocessing.StandardScaler().fit(y_train) # Scaling y not necessary, weights are influenced by differences of x_i
#y_train_scaled = scaler_y.transform(y_train)

#------------------------------------------------------------------------------

l_1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1] # Vector of Ratios to test with CV

alpha_vec = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1] # Vector of different alphas for penalty weighting

elastic_cv = ElasticNetCV(l1_ratio = l_1_ratios, alphas = alpha_vec, cv = 10, n_jobs = -1, random_state = 999, max_iter = 10^7, selection = 'random')
elastic_cv_normalize = ElasticNetCV(l1_ratio = l_1_ratios, normalize = True, alphas = alpha_vec, cv = 10, n_jobs = -1, random_state = 999, max_iter = 10^7, selection = 'random', tol = 1e-8 )



#model_nonregular = elastic_cv.fit(Phi, y_train)
#weights_nonregular = model_nonregular.coef_
#print(model_nonregular.alpha_)
#print(model_nonregular.l1_ratio_)

model_normalized = elastic_cv_normalize.fit(Phi, y_train)
weights_normalized = model_normalized.coef_

#model_phi_regular = elastic_cv.fit(Phi_scaled, y_train)
#weights_phi_regular = model_phi_regular.coef_

#model_x_regular = elastic_cv.fit(Phi_x_scaled, y_train)
#weights_x_regular = model_x_regular.coef_

all_weights = np.vstack((weights_nonregular.T, weights_normalized.T, weights_phi_regular.T, weights_x_regular.T)).T

pd.DataFrame(all_weights).to_csv("weight_comparison.csv",header=None,index=None, float_format = '%.35f')
pd.DataFrame(weights_nonregular).to_csv("weight_nonregular.csv",header=None,index=None, float_format = '%.35f')
pd.DataFrame(weights_normalized).to_csv("weight_normalized.csv",header=None,index=None, float_format = '%.35f')
pd.DataFrame(weights_phi_regular).to_csv("weight_phi_regular.csv",header=None,index=None, float_format = '%.35f')
pd.DataFrame(weights_x_regular).to_csv("weight_x_regular.csv",header=None,index=None, float_format = '%.35f')
