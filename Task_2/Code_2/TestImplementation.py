# Test file to Test Feature Transform_IterativeImp and Subtask1

import FeatureTransform_IterativeImp
import Subtask1
import pandas as pd

train_path = '../Data_2/train_features.csv'

test_path = '../Data_2/test_features.csv'

# Read in Training Labels
train_labels_frame = pd.read_csv('../Data_2/train_labels.csv')

# use of Gradients in Imputer
gradients_active = 1

# Impute Data, return the Pandas Data Frames of Reduced Test Sets with Gradients AND the pure, imputed datasets in the same format as the original ones
train_data_reduced_withGrad_pd, test_data_reduced_withGrad_pd, train_data_imp_pd, test_data_imp_pd = FeatureTransform_IterativeImp.iterativeImpute(train_path, test_path, gradients_active)

# Extract Train Labels for Subtask 1
train_labels_task1_pd = train_labels_frame[['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']]

# Calculate Output for Subtask 1
result_subtask1_pd = Subtask1.solveSubtask1(train_data_reduced_withGrad_pd, test_data_reduced_withGrad_pd, train_labels_task1_pd)