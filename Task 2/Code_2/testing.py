import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import random

this_pid=np.array([[1 ,2,3,4,5,6],[2,np.nan,3,np.nan,4,3],[2,np.nan,3,np.nan,4,3],[5,2,np.nan,np.nan,np.nan,4]])
that_pid=np.array([[6,5,4,3,2,1],[np.nan,4,3,np.nan,9,3],[9,8,0,np.nan,1,5],[2,np.nan,6,np.nan,np.nan,9]])
blabla=np.dstack((this_pid,that_pid))

print(blabla)

print(np.shape(blabla))
print(blabla[:,:,0])
for i in [1]:
    imp_mean=SimpleImputer(missing_values=np.nan,strategy="mean")
    newarr=imp_mean.fit_transform(blabla[:,:,0])
    blabla[:,:,0]=newarr

print(blabla[:,:,0])