#--------------------------------------------------------------------------------------------------
# TASK 1b - Intro to Machine Learning ETHZ SS2021
#
# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#
#--------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn import linear_model
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

# Read in Data
train_data=data('../Data_1b/train.csv',1,2)

x_train = train_data.X[0:600]
y_train = train_data.y[0:600]
x_test = train_data.X[600:]
y_test = train_data.y[600:]

print("xtrain: ", x_train.shape)
print("ytrain: ", y_train.shape)
print("xtest: ", x_test.shape)
print("ytest: ", y_test.shape)

# define features
Phi_train=prepare_data(x_train)
Phi_test = prepare_data(x_test)
#print(Phi_train.shape)
#print(Phi_train[0])

alphas = [0.1, 0.01, 0.001, 0.001, 0.0001, 0.00001, 0.000001]
#tolerances = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
#iterations = [100, 1000, 10000, 100000, 1000000, 10000000]

#dict_a_RMSE = {}
#dict_t_RMSE = {}
dict_i_RMSE = {}

#for a in alphas:
#for t in tolerances:

#for i in iterations:

    # Lasso
#    clf = linear_model.Lasso(alpha=0.01, tol=0.000001, max_iter=i)
#    clf.fit(Phi_train, y_train)
#    ypred = clf.predict(Phi_test)
#    print(clf.coef_)
#    weights = clf.coef_
#    #pd.DataFrame(weights).to_csv("output_pd_"+"lasso"+".csv",header=None,index=None, float_format = '%.35f')
#    RMSE = sqrt(mean_squared_error(y_test, y_pred=ypred))

#    print()
    #dict_a_RMSE[a] = RMSE
    #dict_t_RMSE[t] = RMSE
#    dict_i_RMSE[i] = RMSE

#print('dict, keys = alphas / values = RMSE')
#print('dict, keys = tolerances / values = RMSE')
#print('dict, keys = iterations / values = RMSE')


#print(dict_a_RMSE)
#print(dict_t_RMSE)
#print(dict_i_RMSE)

#print(min(dict_a_RMSE.values()))
#print(min(dict_t_RMSE.values()))
#print(min(dict_i_RMSE.values()))


# Elastic Net


# Lasso

#clf = linear_model.Lasso(alpha=alphas, tol=0.0001, max_iter=100000)
#clf.fit(Phi_train, y_train)
#ypred = clf.predict(Phi_test)
#print(clf.coef_)
#weights = clf.coef_

#pd.DataFrame(weights).to_csv("output_pd_"+"lasso_V2"+".csv",header=None,index=None, float_format = '%.35f')



#LassoCV



'''
# ElasticNet

l_1_ratios = [0,95, 0.99, 0.999, 1] # Vector of Ratios to test with CV

lambdas = [0.1, 0.3, 0.5, 0.7, 1] # Vector of Lambdas for Ridge Regression

elastic_cv = ElasticNetCV(l1_ratio = l_1_ratios, alphas = lambdas, cv = 10, n_jobs=-1, random_state = 42, max_iter = 1000000)
print(Phi_train.shape)


model = elastic_cv.fit(Phi_train, y_train)
weights = model.coef_
print(weights)
#pd.DataFrame(weights).to_csv("output_pd.csv",header=None,index=None, float_format = '%.35f')


#Importieren Sie die Klasse, die die Regressionsmethode enthält.
from sklearn.linear_model import ElasticNet
#Erstellen Sie eine Instanz der Klasse.
EN = ElasticNet(l1_ratio=0.5, alpha= 1)
# alpha ist der Regularisierungsparameter, l1_ratio verteilt Alpha an L1 / L2
# Passen Sie die Instanz an die Daten an und sagen Sie dann den erwarteten Wert voraus.
EN = EN.fit (x_train, y_train)
y_predict = EN.predict(x_test)
#Die ElasticNetCV- Klasse führt eine Kreuzvalidierung für eine Reihe von Werten für l1_ratio und alpha durch.
'''