"""File for global variables. In the cell below some settings can be changed to alter the behaviour during training"""

import torch
from torch import nn

from model_file import VAD_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))
# %% The variables in this cell can be customised
learning_rate = 1e-2 # Learning rate for the EB, FB and DB layers
learning_rate_DN = 1e-3 # Learning rate for the discriminative network
LR_factor = 0.7 # The factor with which to decrease the learning rate after each epoch

training_epochs = 30 # The number of epochs during training
concatenates = 10 # The number of files to concatenate
training_batch_size = 3  # The number of forward steps per backward step. Multiplied with "concatenates" this is the mini-batch size
testing_batch_size = 650 # The number of files in the testing split. Validation split is half of this
files_per_epoch = 8400
AN_weight = 0.1 # The scalar referred to as "alpha" in the paper

output_folder = "Enter your folder name here" # The name of the folder in which to store the results and models

training_data_path = r"your path\aurora2\SPEECHDATA\\TRAIN"
training_label_path = r"your path\Aurora2TrainSet-ReferenceVAD"

testing_data_path = r"your path\aurora2\SPEECHDATA\TESTB"
testing_label_path = r"your path\Aurora2TestSet-ReferenceVAD\\"

"""Kernel sizes"""
k_EB1 = 55
k_EB2 = 160
k_EB3 = 160
k_EB4 = 160

k_FB = 160

k_DB1 = 55
k_DB2 = 15
k_DB3 = 5

k_DN1 = 55
k_DN2 = 15
k_DN3 = 5

# %%
training = 0 # Flag denoting whether the model is being trained or tested
validation = 0 # Flag denoting whether to use the testing or validation split

# Initialize the loss function
loss_primary = nn.BCELoss()
loss_secondary = nn.CrossEntropyLoss()

# Make an instance of the model
VAD = VAD_model().to(device)

""" Initialize the optimisers"""
optimizer_EB1 = torch.optim.RMSprop(VAD.EB1.parameters(), lr=learning_rate)
optimizer_EB2 = torch.optim.RMSprop(VAD.EB2.parameters(), lr=learning_rate)
optimizer_EB3 = torch.optim.RMSprop(VAD.EB3.parameters(), lr=learning_rate)
optimizer_EB4 = torch.optim.RMSprop(VAD.EB4.parameters(), lr=learning_rate)

optimizer_FB = torch.optim.RMSprop(VAD.FB.parameters(), lr=learning_rate)

optimizer_DN1 = torch.optim.RMSprop(VAD.DN1.parameters(), lr=learning_rate_DN)
optimizer_DN2 = torch.optim.RMSprop(VAD.DN2.parameters(), lr=learning_rate_DN)
optimizer_DN3 = torch.optim.RMSprop(VAD.DN3.parameters(), lr=learning_rate_DN)

optimizer_DB1 = torch.optim.RMSprop(VAD.DB1.parameters(), lr=learning_rate)
optimizer_DB2 = torch.optim.RMSprop(VAD.DB2.parameters(), lr=learning_rate)
optimizer_DB3 = torch.optim.RMSprop(VAD.DB3.parameters(), lr=learning_rate)




noise_type_AURORA = "caf"
SNR_level_AURORA = 0


""" Storing results from training and validation"""
training_results_big = {
    "training" : [],
    "val_caf_-5" : [],
    "val_caf_0" : [],
    "val_caf_5" : [],
    "val_caf_10" : [],
    "val_caf_15" : [],
    "val_caf_20" : [],
    "val_caf_CLEAN" : [],
    "test_caf_-5" : [],
    "test_caf_0" : [],
    "test_caf_5" : [],
    "test_caf_10" : [],
    "test_caf_15" : [],
    "test_caf_20" : [],
    "test_caf_CLEAN" : [],
    "val_bbl_-5" : [],
    "val_bbl_0" : [],
    "val_bbl_5" : [],
    "val_bbl_10" : [],
    "val_bbl_15" : [],
    "val_bbl_20" : [],
    "val_bbl_CLEAN" : [],
    "test_bbl_-5" : [],
    "test_bbl_0" : [],
    "test_bbl_5" : [],
    "test_bbl_10" : [],
    "test_bbl_15" : [],
    "test_bbl_20" : [],
    "test_bbl_CLEAN" : [],
    "val_bus_-5" : [],
    "val_bus_0" : [],
    "val_bus_5" : [],
    "val_bus_10" : [],
    "val_bus_15" : [],
    "val_bus_20" : [],
    "val_bus_CLEAN" : [],
    "test_bus_-5" : [],
    "test_bus_0" : [],
    "test_bus_5" : [],
    "test_bus_10" : [],
    "test_bus_15" : [],
    "test_bus_20" : [],
    "test_bus_CLEAN" : [],
    "val_ped_-5" : [],
    "val_ped_0" : [],
    "val_ped_5" : [],
    "val_ped_10" : [],
    "val_ped_15" : [],
    "val_ped_20" : [],
    "val_ped_CLEAN" : [],
    "test_ped_-5" : [],
    "test_ped_0" : [],
    "test_ped_5" : [],
    "test_ped_10" : [],
    "test_ped_15" : [],
    "test_ped_20" : [],
    "test_ped_CLEAN" : [],
    "val_ssn_-5" : [],
    "val_ssn_0" : [],
    "val_ssn_5" : [],
    "val_ssn_10" : [],
    "val_ssn_15" : [],
    "val_ssn_20" : [],
    "val_ssn_CLEAN" : [],
    "test_ssn_-5" : [],
    "test_ssn_0" : [],
    "test_ssn_5" : [],
    "test_ssn_10" : [],
    "test_ssn_15" : [],
    "test_ssn_20" : [],
    "test_ssn_CLEAN" : [],
    "val_str_-5" : [],
    "val_str_0" : [],
    "val_str_5" : [],
    "val_str_10" : [],
    "val_str_15" : [],
    "val_str_20" : [],
    "val_str_CLEAN" : [],
    "test_str_-5" : [],
    "test_str_0" : [],
    "test_str_5" : [],
    "test_str_10" : [],
    "test_str_15" : [],
    "test_str_20" : [],
    "test_str_CLEAN" : [],
    "epochs" : [],
    "learning_rate" : [],
    "loss_DB" : [],
    "loss_AN" : [],
    "test_TP" : [],
    "test_FP" : [],
    "test_TN" : [],
    "test_FN" : [],
    "time_passed" : []
    }

""" Storing results from testing"""
training_results_AUC = {
   
    "N1_-5_ACC" : [],
    "N1_0_ACC" : [],
    "N1_5_ACC" : [],
    "N1_10_ACC" : [],
    "N1_15_ACC" : [],
    "N1_20_ACC" : [],
    "N1_CLEA_ACC" : [],
    
    "N1_-5_TP" : [],
    "N1_-5_FP" : [],
    "N1_-5_TN" : [],
    "N1_-5_FN" : [],
    "N1_0_TP" : [],
    "N1_0_FP" : [],
    "N1_0_TN" : [],
    "N1_0_FN" : [],
    "N1_5_TP" : [],
    "N1_5_FP" : [],
    "N1_5_TN" : [],
    "N1_5_FN" : [],
    "N1_10_TP" : [],
    "N1_10_FP" : [],
    "N1_10_TN" : [],
    "N1_10_FN" : [],
    "N1_15_TP" : [],
    "N1_15_FP" : [],
    "N1_15_TN" : [],
    "N1_15_FN" : [],
    "N1_20_TP" : [],
    "N1_20_FP" : [],
    "N1_20_TN" : [],
    "N1_20_FN" : [],
    "N1_CLEA_TP" : [],
    "N1_CLEA_FP" : [],
    "N1_CLEA_TN" : [],
    "N1_CLEA_FN" : [],
    
    "N2_-5_ACC" : [],
    "N2_0_ACC" : [],
    "N2_5_ACC" : [],
    "N2_10_ACC" : [],
    "N2_15_ACC" : [],
    "N2_20_ACC" : [],
    "N2_CLEA_ACC" : [],
    
    "N2_-5_TP" : [],
    "N2_-5_FP" : [],
    "N2_-5_TN" : [],
    "N2_-5_FN" : [],
    "N2_0_TP" : [],
    "N2_0_FP" : [],
    "N2_0_TN" : [],
    "N2_0_FN" : [],
    "N2_5_TP" : [],
    "N2_5_FP" : [],
    "N2_5_TN" : [],
    "N2_5_FN" : [],
    "N2_10_TP" : [],
    "N2_10_FP" : [],
    "N2_10_TN" : [],
    "N2_10_FN" : [],
    "N2_15_TP" : [],
    "N2_15_FP" : [],
    "N2_15_TN" : [],
    "N2_15_FN" : [],
    "N2_20_TP" : [],
    "N2_20_FP" : [],
    "N2_20_TN" : [],
    "N2_20_FN" : [],
    "N2_CLEA_TP" : [],
    "N2_CLEA_FP" : [],
    "N2_CLEA_TN" : [],
    "N2_CLEA_FN" : [],
    
    "N3_-5_ACC" : [],
    "N3_0_ACC" : [],
    "N3_5_ACC" : [],
    "N3_10_ACC" : [],
    "N3_15_ACC" : [],
    "N3_20_ACC" : [],
    "N3_CLEA_ACC" : [],
    
    "N3_-5_TP" : [],
    "N3_-5_FP" : [],
    "N3_-5_TN" : [],
    "N3_-5_FN" : [],
    "N3_0_TP" : [],
    "N3_0_FP" : [],
    "N3_0_TN" : [],
    "N3_0_FN" : [],
    "N3_5_TP" : [],
    "N3_5_FP" : [],
    "N3_5_TN" : [],
    "N3_5_FN" : [],
    "N3_10_TP" : [],
    "N3_10_FP" : [],
    "N3_10_TN" : [],
    "N3_10_FN" : [],
    "N3_15_TP" : [],
    "N3_15_FP" : [],
    "N3_15_TN" : [],
    "N3_15_FN" : [],
    "N3_20_TP" : [],
    "N3_20_FP" : [],
    "N3_20_TN" : [],
    "N3_20_FN" : [],
    "N3_CLEA_TP" : [],
    "N3_CLEA_FP" : [],
    "N3_CLEA_TN" : [],
    "N3_CLEA_FN" : [],
    
    "N4_-5_ACC" : [],
    "N4_0_ACC" : [],
    "N4_5_ACC" : [],
    "N4_10_ACC" : [],
    "N4_15_ACC" : [],
    "N4_20_ACC" : [],
    "N4_CLEA_ACC" : [],
    
    "N4_-5_TP" : [],
    "N4_-5_FP" : [],
    "N4_-5_TN" : [],
    "N4_-5_FN" : [],
    "N4_0_TP" : [],
    "N4_0_FP" : [],
    "N4_0_TN" : [],
    "N4_0_FN" : [],
    "N4_5_TP" : [],
    "N4_5_FP" : [],
    "N4_5_TN" : [],
    "N4_5_FN" : [],
    "N4_10_TP" : [],
    "N4_10_FP" : [],
    "N4_10_TN" : [],
    "N4_10_FN" : [],
    "N4_15_TP" : [],
    "N4_15_FP" : [],
    "N4_15_TN" : [],
    "N4_15_FN" : [],
    "N4_20_TP" : [],
    "N4_20_FP" : [],
    "N4_20_TN" : [],
    "N4_20_FN" : [],
    "N4_CLEA_TP" : [],
    "N4_CLEA_FP" : [],
    "N4_CLEA_TN" : [],
    "N4_CLEA_FN" : [],
    
    "time_passed" : [],
    "threshold" : [],
    "alpha" : []
    }


