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
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
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
np.random.seed(999)
# Read in Data
train_data=data('../Data_1b/train.csv',1,2)
x_train = train_data.X

###
# normalization of X
x_train = preprocessing.normalize(x_train, norm='l2')
###

y_train = train_data.y

scaler = preprocessing.StandardScaler()
x_train_regularized = scaler.fit_transform(x_train)

Phi=prepare_data(x_train) # Regressor Matrix non regularized X

kernel = np.matmul(Phi, Phi.T)
X = kernel # Kernel Matrix for Regression

# Cross Validation
list_1 = list(np.linspace(50, 100, 1000))
list_2 = [1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.525, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1, 9, 1001, 10000]
list_comb = list_1 + list_2
param_grid = {"alpha": list_comb}

CV = GridSearchCV(KernelRidge(kernel = "precomputed"), param_grid = param_grid, scoring = 'neg_mean_absolute_error', cv = 175, n_jobs = -1)

CV.fit(X, y_train)

results = CV.cv_results_

best_est = CV.best_estimator_
print('best est: ', best_est)
best_params = CV.best_params_
print('best params: ', best_params)

# coeff_direct = np.dot(np.linalg.pinv(kernel), y_train)
coeff_class = best_est.dual_coef_

# w_direct = np.dot(Phi.T, coeff_direct)
w_class = np.dot(Phi.T, coeff_class)

#pd.DataFrame(w_direct).to_csv("kernel_directRegression_noRegularization_RawX.csv",header=None,index=None, float_format = '%.35f')
pd.DataFrame(w_class).to_csv("kernel_Ridge_GridSearchCV_175fold_rawX_normalized_scoreRMSE.csv",header=None,index=None, float_format = '%.50f')

pd.DataFrame(results).to_csv("results_V3.csv")