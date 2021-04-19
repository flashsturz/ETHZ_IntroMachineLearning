# Janik Baumer, Flavio Regenass, Simon Tobler
# jbaumer, flavior, sitobler
#-----------------
# Description Task 2

#--------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import random
import time as time

def md_list2pdSeries(inputlist,pd_col_list,name):
    [n_times,n_meas,n_pid]=np.shape(inputlist)
    #print(n_times) #for debugging
    #print(n_meas) #for debugging
    #print(n_pid) #for debugging
    stackedList = inputlist[:, :, 0]
    for pid in range(1,n_pid):
        stackedList=np.concatenate((stackedList,inputlist[:,:,pid]),axis=0)
    #print(np.shape(stackedList)) #for debugging

    pd_dataframe=pd.DataFrame(data=stackedList,columns=pd_col_list)

    filename="test_features_simpleImpute_"+name+".csv"
    pd_dataframe.to_csv(filename,index=False)

def print_elapsed_time(starttime):
    time_now=time.perf_counter()
    elapsed_time=time_now-starttime
    print("Time elapsed since start: %.2f s" % elapsed_time)

#--------------------------------------------------------------------------------------------------
strat='median'


print("Starts. Reading in Data...")
totaltime_start=time.perf_counter()

fulldata=pd.read_csv('Data_2/test_features.csv')

random.seed(1234)

#Writing full dataset to multidimensional np array:
md_list=np.array([])

#----------
print("Preparing prerequisits...")
print_elapsed_time(totaltime_start)
#----------

list_pid=fulldata.pid.unique()
md_list=fulldata.loc[fulldata['pid'] == list_pid[0]].to_numpy()
list_pid=np.delete(list_pid,0)
for pid in list_pid:
    this_data = fulldata.loc[fulldata['pid'] == pid].to_numpy()
    md_list=np.dstack((md_list,this_data))
[n_times,n_meas,n_pid]=np.shape(md_list)

#Find averages over all patients for each measurement to get a bias value
#   (needed to perform meadian or mean imputing in cols full of nan)
fulldata_np=fulldata.to_numpy()
#print(np.shape(fulldata_np)) #for debugging
(fulldata_rows, fulldata_cols)=np.shape(fulldata_np)
avg_fulldata=np.nanmean(fulldata_np,axis=0)

md_list_impmean = md_list
md_list_impmedian = md_list
md_list_impconst = md_list

print("Finished prep. Imputing starts...")
print_elapsed_time(totaltime_start)

#for pid in [0]:
for pid in range(n_pid):
    this_pid = md_list[:,:,pid]

    if (strat!='constant'):
        #Check if there is a col with all nan-values:
        isnan_bool=np.all(np.isnan(this_pid),axis=0)
        for isnan_col in range(np.shape(isnan_bool)[0]):
            if isnan_bool[isnan_col]:
                #If all nan: isnan_col is true thus set one entry of col to avg over all patients:
                this_pid[2,isnan_col]=avg_fulldata[isnan_col]

    #print(np.shape(this_pid)) #for debugging

    #Imputers:
    imp_mean = SimpleImputer(missing_values=np.nan,strategy=strat)

    this_pid_impmean=imp_mean.fit_transform(this_pid)
    #print(this_pid_impmean)

    #write imputed Persondata to the muldidimensional list
    #print(np.shape(md_list_impmean))
    #print(np.shape(this_pid_impmean))
    md_list_impmean[:,:,pid]=this_pid_impmean

#print(md_list_impmean[:, 4:9, pid])
print("Imputing finished, starting to write the multidimensional lists to csv's...")
print_elapsed_time(totaltime_start)


md_list2pdSeries(md_list_impmean,fulldata.columns.tolist(),strat)
print("Finished writing imputed Data to file...")
print_elapsed_time(totaltime_start)

print("Finished Execution, new Datasets in correspondig Files.")
print_elapsed_time(totaltime_start)