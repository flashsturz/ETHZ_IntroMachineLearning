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
from sklearn.model_selection import GridSearchCV, validation_curve, RepeatedKFold
from math import sqrt
import matplotlib.pyplot as plt

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
# Decided if normalized X or no --> no normalization according to Piazza
normalizationX = 0
# Read in Data
train_data=data('../Data_1b/train.csv',1,2)
x_train = train_data.X
y_train = train_data.y

pd_train_data = pd.read_csv('../Data_1b/train.csv')
pd_train_data_x = pd_train_data[["x1", "x2", "x3", "x4", "x5"]]
pd_train_data_x_regularized = preprocessing.StandardScaler().fit_transform(pd_train_data_x)


scaler = preprocessing.StandardScaler()
x_train_regularized = scaler.fit_transform(x_train)

if normalizationX == 1:
    Phi = prepare_data(x_train_regularized) # Regressor Matrix with regularized X
else:
    Phi=prepare_data(x_train) # Regressor Matrix non regularized X

kernel = np.matmul(Phi, Phi.T)
X = kernel # Kernel Matrix for Regression

# Cross Validation
if normalizationX == 1:
    list_1 = list(np.linspace(10, 40, 500))
else:
    list_1 = list(np.linspace(10, 60, 1000))

list_2 = [1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.525, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1, 9, 1001, 10000]
list_comb = list_1 + list_2
list_comb.sort()
param_grid = {"alpha": list_comb}
cv_ = 10
repeats = 10
cv_obj = RepeatedKFold(n_splits=cv_, n_repeats=repeats, random_state=999)

#train_scores, valid_scores = validation_curve(KernelRidge(kernel = "precomputed"), X, y_train, param_name = "alpha", param_range = list_comb, cv = cv_, scoring = "neg_root_mean_squared_error", n_jobs = -1)
#train_scores_mean = np.mean(train_scores, axis=1)
#train_scores_std = np.std(train_scores, axis=1)
#test_scores_mean = np.mean(valid_scores, axis=1)
#test_scores_std = np.std(valid_scores, axis=1)
#plt.title("Validation Curve with Kernel Ridge")
#plt.xlabel(r"$\lambda$")
#plt.ylabel("Score")
#plt.ylim(-2.2, -1.6)
#lw = 2
#plt.semilogx(list_comb, train_scores_mean, label="Training score",
#             color="darkorange", lw=lw)
#plt.fill_between(list_comb, train_scores_mean - train_scores_std,
#                 train_scores_mean + train_scores_std, alpha=0.2,
#                 color="darkorange", lw=lw)
#plt.semilogx(list_comb, test_scores_mean, label="Cross-validation score",
#             color="navy", lw=lw)
#plt.fill_between(list_comb, test_scores_mean - test_scores_std,
#                 test_scores_mean + test_scores_std, alpha=0.2,
#                 color="navy", lw=lw)
#plt.legend(loc="best")
#plt.show()


CV = GridSearchCV(KernelRidge(kernel = "precomputed"), param_grid = param_grid, scoring = 'neg_root_mean_squared_error', cv = cv_obj, n_jobs = -1)

CV.fit(X, y_train)

results = CV.cv_results_

best_est = CV.best_estimator_
best_params = CV.best_params_

# coeff_direct = np.dot(np.linalg.pinv(kernel), y_train)
coeff_class = best_est.dual_coef_

# w_direct = np.dot(Phi.T, coeff_direct)
w_class = np.dot(Phi.T, coeff_class)

#pd.DataFrame(w_direct).to_csv("kernel_directRegression_noRegularization_RawX.csv",header=None,index=None, float_format = '%.35f')
pd.DataFrame(w_class).to_csv("kernel_Ridge_GridSearchCV_10foldRepCV_rawX_scoreRMSE.csv",header=None,index=None, float_format = '%.50f')

pd.DataFrame(results).to_csv("results_repCV.csv")