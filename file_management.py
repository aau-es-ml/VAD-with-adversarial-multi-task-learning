"""File for loading and saving results and models"""

import pickle
import torch
import os
import config

folder = config.output_folder
def save_model(models,t):
    """
    Function for saving the model after each training epoch
    """
    current_dir = os.getcwd()
    res_path = f"{current_dir}\\results"
    fpath = f"{res_path}\\{folder}"
    fname = f"{fpath}\\{t+1}_epochs.pth"
    try:
        os.mkdir(res_path)
    except OSError:
        pass
    try:
        os.mkdir(fpath)
    except OSError:
        pass
    torch.save(config.VAD.state_dict(), fname)
    
def save_model_initial(models):
    """
    Function for saving a model that can be used as a common initialisation
    """
    current_dir = os.getcwd()
    res_path = f"{current_dir}\\results"
    fname = f"{res_path}\\initial4.pth"
    try:
        os.mkdir(res_path)
    except OSError:
        pass
    else:
        print ("Successfully created the directory %s " % res_path)
    torch.save(config.VAD.state_dict(), fname)
    
def save_results(res,t):
    """
    Function for saving the dictionary containing information on training and validation as a pickle file
    """
    current_dir = os.getcwd()
    res_path = f"{current_dir}\\results"
    fpath = f"{res_path}\\{folder}"
    fname = f"{fpath}\\epoch_{t}.pickle"
    try:
        os.mkdir(res_path)
    except OSError:
        pass
    try:
        os.mkdir(fpath)
    except OSError:
        pass
    
    with open(fname, "wb") as fp:   #Pickling
        pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)    
        
def save_results_AUC(res, t):
    """
    Function for saving the dictionary containing information on testing as a pickle file
    """
    current_dir = os.getcwd()
    res_path = f"{current_dir}\\results"
    fpath = f"{res_path}\\{folder}"
    fname = f"{fpath}\\AUC_results_set{config.dset}_epoch{t}.pickle"
    try:
        os.mkdir(res_path)
    except OSError:
        pass
    try:
        os.mkdir(fpath)
    except OSError:
        pass
    with open(fname, "wb") as fp:   #Pickling
        pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)  

def load_results():
    """
    Function for loading a saved pickle file
    """
    current_dir = os.getcwd()
    with open(f"{current_dir}\\results\\epoch.pickle", "rb") as input_file:
        return pickle.load(input_file)
    
def load_model():
    """
    Function for loading a saved model
    """
    current_dir = os.getcwd()
    fpath = f"{current_dir}"
    # fname = f"{fpath}\\model.pth"
    fname = f"{fpath}\\results\\batchnorm\\20_epochs.pth"
    fname = r"C:\Users\claus\OneDrive - Aalborg Universitet\kode\hopefully final\smukkeficeret kode\results\batchnorm2\20_epochs.pth"
    config.VAD.load_state_dict(torch.load(fname), strict=False)
    return config.VAD
    