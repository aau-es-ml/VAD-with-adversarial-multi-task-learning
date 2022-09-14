"""File for performing tests on the validation and testing set"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from training import calc_loss, calc_loss_noisetypes


    
def make_plots(preds, X, y, labs):
    """
    Plots the raw waveform, scores of speech and non-speech, VAD predictions and true VAD labels
    """
    y = y[0:len(labs)]
    plt.plot(X[0:len(preds[0,:])*80]/max(X)*0.5)
    samples_p_label = np.ones((80,))
    plt.plot(np.kron(preds[0,:], samples_p_label)-0.5,'g')
    plt.plot(np.kron(preds[1,:], samples_p_label)-0.5,'r')
    plt.plot(np.kron(y, samples_p_label)-0.5,'b')
    plt.plot(np.kron(labs, samples_p_label))
    
    plt.plot(np.kron(abs(labs-y), samples_p_label)-1.6)
    new_acc = {1-sum((labs-y!=0))/len(y)}
    plt.ylim([-1.1,1.1])
    plt.title(f"Noise type: {config.noise_type_AURORA} - SNR level: {config.SNR_level_AURORA} - acc: {new_acc}")
    plt.show
    plt.pause(0.0005)


def calc_pos_neg(labels, predictions):
    """
    Calculates the number of true positive, false positives, true negatives and false negatives in the last forward step
    """
    labels = labels[0:len(predictions)]
    predictions_inv = (predictions-1)*-1;
    labels_inv = (labels-1)*-1;
    TP = sum(predictions[labels[0:len(predictions)]==1])
    FP = sum(predictions[labels[0:len(predictions)]==0])
    FN = sum(predictions_inv[labels[0:len(predictions)]==1])
    TN = sum(predictions_inv[labels_inv[0:len(predictions)]==1])
    return TP, FP, TN, FN
        
def validation_loop(test_data_loader, t):
    """Variable initialisations"""
    accumulated_acc = 0
    last_batch = 0
    concats = config.concatenates
    
    """ Initialises empty tensors for the data to be concatenated"""
    concat_X = torch.empty((1,1,1,0), device = config.device)
    concat_y = torch.empty((1,0), device = config.device)
    
    """Initialises variables to store information about predictions"""
    TP_acc = 0; FP_acc = 0; TN_acc = 0; FN_acc = 0;
    times = 0
    files = 0
    for batch, (X, y) in enumerate(test_data_loader):
        y = y.to(config.device)
        X = X.to(config.device)
        
        """Makes sure the length of y and X corresponds"""
        y_length = np.floor(len(X[0,0,0,:])/80)
        xmax = min(len(y[0,:])*80, y_length*80)
        y = y[:,0:int(y_length)]
        X = X[:,:,:,0:int(xmax)]
        
        """Concatenate audio and VAD labels"""
        if len(X[0,0,0,:]) != 0:
            files += 1
            concat_X = torch.cat((concat_X, X),3)
            concat_y = torch.cat((concat_y, y),1)

            """The main validation loop. Runs after sufficient files are concatenated"""
            if files > last_batch + concats:
                last_batch = files
                """Stores the data in original variable names and resets the tensors containing the concatenated files"""
                X = concat_X
                y = concat_y
                
                concat_X = torch.empty((1,1,1,0), device = config.device)
                concat_y = torch.empty((1,0), device = config.device)
                
                """Forward step"""
                pred, placeholder = config.VAD(X[0,:,:,:].float())
                
                """Calculates VAD predictions"""
                labs = (pred[0,0,:]>pred[0,1,:]).to('cpu').detach().numpy()
                npy = y[0,:].to('cpu').detach().numpy()
                npy = npy[0:len(labs)]
                acc = 1-sum((labs-npy!=0))/len(npy)
                accumulated_acc += acc
                times = times + 1
                
                """ Calculates  the number of true positives, false positive etc."""
                TP, FP, TN, FN = calc_pos_neg(npy, labs)
                TP_acc += TP
                FP_acc += FP
                TN_acc += TN
                FN_acc += FN
                
                """When predictions of all files have been computed, the data is saved"""
                if (files+1) > (config.testing_batch_size - (config.testing_batch_size*(config.validation*0.5))):
                    
                    """Calculates the accuracy before plotting the raw waveform, the speech and non-speech scores and the predicted and true VAD labels"""
                    batch_accuracy = accumulated_acc/(times)
                    print(f"Accuracy of batch: {batch_accuracy}\n")
                    npX = X[0,0,0,:].to('cpu').detach().numpy()
                    npy = y[0,:].to('cpu').detach().numpy()
                    preds_np = pred[0,:,:].to('cpu').detach().numpy()
                    make_plots(preds_np, npX, npy, labs)
    
                    total_samples = FP_acc + TN_acc + TP_acc + FN_acc # Used to normalise the data
                    
                    """Saves data to dictionary"""
                    config.training_results_big["test_TP"].append(TP_acc/total_samples)
                    config.training_results_big["test_FP"].append(FP_acc/total_samples)
                    config.training_results_big["test_TN"].append(TN_acc/total_samples)
                    config.training_results_big["test_FN"].append(FN_acc/total_samples)
                             
                    return


def testing_loop(test_data_loader, t):
    """Variable initialisations"""
    ROC_samples = 51 # The number of points in the ROC curve
    last_batch = 0
    concats = config.concatenates
    concat_X = torch.empty((1,1,1,0), device = config.device)
    concat_y = torch.empty((1,0), device = config.device)
    TP_acc = np.zeros((ROC_samples,))
    FP_acc = np.zeros((ROC_samples,))
    TN_acc = np.zeros((ROC_samples,))
    FN_acc = np.zeros((ROC_samples,))
    accumulated_acc = np.zeros((ROC_samples,))
    files = 0
    loss = 0
    for batch, (X, y) in enumerate(test_data_loader):
        y = y.to(config.device)
        X = X.to(config.device)
        
        """Makes sure the length of y and X corresponds"""
        y_length = np.floor(len(X[0,0,0,:])/80)
        xmax = min(len(y[0,:])*80, y_length*80)
        y = y[:,0:int(y_length)]
        X = X[:,:,:,0:xmax]
        
        """Concatenate audio and VAD labels"""
        if len(X[0,:,0,0]) != 0:
            files += 1
            concat_X = torch.cat((concat_X, X),3)
            concat_y = torch.cat((concat_y, y),1)
            
            """The main testing loop. Runs after sufficient files are concatenated"""
            if files > last_batch + concats:
                last_batch = files
                
                """Stores the data in original variable names and resets the tensors containing the concatenated files"""
                X = concat_X
                y = concat_y
                
                concat_X = torch.empty((1,1,1,0), device = config.device)
                concat_y = torch.empty((1,0), device = config.device)
                
                """Forward step"""
                pred, placeholder = config.VAD(X[0,:,:,:].float())
                labs = []
                acc = []
                
                loss_DB = calc_loss(y,pred)
                loss += loss_DB.item()
                del loss_DB
                
                """Sweeps over the the scores of speech and non-speech channels using different thresholds and generates predictions"""
                for index, (threshold) in enumerate(np.linspace(-1,1,ROC_samples)):
                    labs = (pred[0,0,:]>pred[0,1,:]+threshold).to('cpu').detach().numpy()
                    npy=y[0,:].to('cpu').detach().numpy()
                    npy = npy[0:len(labs)]
                    acc.append(1-sum((labs-npy!=0))/len(npy))
                    accumulated_acc[index] += acc[-1]
                    
                    """ Calculates  the number of true positives, false positive etc."""
                    TP, FP, TN, FN = calc_pos_neg(npy, labs)
                    
                    """Each index corresponds to a threshold value"""
                    TP_acc[index] += TP
                    FP_acc[index] += FP
                    TN_acc[index] += TN
                    FN_acc[index] += FN
                    if (files+1) > int(config.testing_batch_size - (config.testing_batch_size*(config.validation*0.5))):
                        """Calculates the accuracy before plotting the raw waveform, the speech and non-speech scores and the predicted and true VAD labels"""
                        npX = X[0,0,0,:].to('cpu').detach().numpy()
                        npy = y[0,:].to('cpu').detach().numpy()
                        preds = pred[0,:,:].to('cpu').detach().numpy()
                        make_plots(preds, npX, npy, labs)
                        
                        """Saves the data into a dictionary"""
                        dict_key_PN = f"{config.noise_type_AURORA}_{config.SNR_level_AURORA}" 
                        total_samples = FP_acc + TN_acc + TP_acc + FN_acc # Used to normalise the data
                        config.training_results_AUC[f"{dict_key_PN}_TP"] = (TP_acc/total_samples)
                        config.training_results_AUC[f"{dict_key_PN}_FP"] = (FP_acc/total_samples)
                        config.training_results_AUC[f"{dict_key_PN}_TN"] = (TN_acc/total_samples)
                        config.training_results_AUC[f"{dict_key_PN}_FN"] = (FN_acc/total_samples)              
                        config.training_results_AUC[f"{dict_key_PN}_loss_VAD"] = loss                       
                        return
    
