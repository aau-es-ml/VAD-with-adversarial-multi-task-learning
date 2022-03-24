"""File for training the model"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import config

def update_learning_rates():
    """
    Updates the learning rate of all layers"
    """
    for param_group in config.optimizer_EB1.param_groups:
        param_group['lr'] = config.learning_rate
    for param_group in config.optimizer_EB2.param_groups:
        param_group['lr'] = config.learning_rate
    for param_group in config.optimizer_EB3.param_groups:
        param_group['lr'] = config.learning_rate
    for param_group in config.optimizer_EB4.param_groups:
        param_group['lr'] = config.learning_rate
    for param_group in config.optimizer_FB.param_groups:
        param_group['lr'] = config.learning_rate
    for param_group in config.optimizer_DB3.param_groups:
        param_group['lr'] = config.learning_rate
    for param_group in config.optimizer_DB1.param_groups:
        param_group['lr'] = config.learning_rate
    for param_group in config.optimizer_DB2.param_groups:
        param_group['lr'] = config.learning_rate
    for param_group in config.optimizer_DN3.param_groups:
        param_group['lr'] = config.learning_rate_DN*-1
    for param_group in config.optimizer_DN1.param_groups:
        param_group['lr'] = config.learning_rate_DN*-1
    for param_group in config.optimizer_DN2.param_groups:
        param_group['lr'] = config.learning_rate_DN*-1


def calc_loss(true_labels, predictions):
    """
    Calculates the loss from the VAD output
    """
    labels_two_channels = np.zeros((2,len(true_labels[0,:])))
    labels_two_channels = torch.from_numpy(labels_two_channels).to(config.device)
    index_min = min(len(true_labels[0,:]),len(predictions[0,0,:]))
    labels_two_channels[0,:] = true_labels
    labels_two_channels[1,:] = 1-true_labels
    labels_two_channels = labels_two_channels[:,0:index_min]
    
    loss = config.loss_primary(predictions[0,:,0:index_min].T.float(),labels_two_channels[:,:].T.float())
    return loss

def calc_loss_noisetypes(noise_true, noise_predicted):
    """
    Calculates the loss from the discriminative network
    """
    noise_true = torch.reshape(noise_true, (-1,))
    loss = config.loss_secondary(noise_predicted.T, noise_true.long())   
    return loss

def back_propagation_full(loss, t):
    """
    Performs the backward step to calculate gradients, then updates the parameters
    """

    config.optimizer_EB1.zero_grad() 
    config.optimizer_EB2.zero_grad() 
    config.optimizer_EB3.zero_grad() 
    config.optimizer_EB4.zero_grad() 
    config.optimizer_FB.zero_grad() 
    config.optimizer_DB1.zero_grad() 
    config.optimizer_DB2.zero_grad() 
    config.optimizer_DB3.zero_grad() 
    config.optimizer_DN1.zero_grad() 
    config.optimizer_DN2.zero_grad() 
    config.optimizer_DN3.zero_grad() 
    loss.backward()

    config.optimizer_EB1.step()
    config.optimizer_EB2.step()
    config.optimizer_EB3.step()
    config.optimizer_EB4.step()

    config.optimizer_FB.step()
    config.optimizer_DB1.step()
    config.optimizer_DB2.step()
    config.optimizer_DB3.step()
    
    config.optimizer_DN1.step()
    config.optimizer_DN2.step()
    config.optimizer_DN3.step()
          

def after_batches(batch_size, accumulated_accuracy, loss_acc, batch, loss, loss_AN, X, accuracy, size):
    """
    Prints various information in the console after each backward step
    """
    accumulated_accuracy /= batch_size
    loss, current = loss.item(), batch * len(X)
    loss_acc /= batch_size

    print(f"Current file/Number of files: [{current+1:>5d}/{size:>5d}]")
    print(f"loss VAD: {loss_acc:>7f}")
    print(f"Loss DN: {loss_AN}")
    print(f"Accuracy of batch: {accumulated_accuracy*100:>4f}")
    print(f"Learning rate: {config.learning_rate:>4f}\n")
    

def calc_accuracy(pred, y):
    """
    Finds the predicted labels by comparing the speech and non-speech channels, then calculates the accuracy. Returns both
    """
    labs = (pred[:,0,0]>pred[:,1,0]).to('cpu').detach().numpy()
    y = y[:,0:len(labs)]
    accuracy = 1-sum((abs(labs-y[0,:].to('cpu').detach().numpy())))/len(y[0,:].to('cpu').detach().numpy())
    return accuracy, labs
    
def make_plots(predictions, X, y, true_labels):
    """
    Plots the raw waveform, scores of speech and non-speech, VAD predictions and true VAD labels
    """
    npX = X[0,0,0,:].to('cpu').detach().numpy()
    npy = y[0,:].to('cpu').detach().numpy()
    
    plt.plot(np.linspace(0,len(predictions[:,0]),len(npX)), (npX)/max((npX*2)))
    plt.plot(predictions[:,0]-0.5,'g')
    plt.plot(predictions[:,1]-0.5,'r')    
    plt.plot(npy-0.5,'b')    
    plt.plot(true_labels)
    plt.ylim([-1,1.1])
    plt.show
    plt.pause(0.0005)


            

def train_loop(train_data_loader, t):
    """Variable initialisations"""
    size = len(train_data_loader.dataset)
    total_acc = 0
    total_loss_DB = 0
    total_loss_AN = 0
    loss_acc = 0
    loss_AN_acc = 0
    accumulated_accuracy = 0
    comb_loss = 0
    last_batch = 0
    concats = config.concatenates # The number of files to concatenate
    
    """ Initialises empty tensors for the data to be concatenated"""
    concat_X = torch.empty((1,1,1,0), device = config.device)
    concat_y = torch.empty((1,0), device = config.device)
    concat_noise = torch.empty((1,0), device = config.device)
    
    for batch, (X, y, noise_true) in enumerate(train_data_loader):
        
        """Initialise the loss if it is not already"""
        try:
            comb_loss
        except NameError:
            comb_loss = 0
        y = y.to(config.device)
        X = X.to(config.device)

        """Creates a tensor of similar size to y containing the true labels for noise type"""
        noises = ["N1", "N2", "N3", "N4", "CL"] 
        noise_index = noises.index(''.join(noise_true))
        noise_index = noise_index
        noise_vector_ini = np.zeros(np.shape(y)) + noise_index
        noise_vector_ini = torch.from_numpy(noise_vector_ini).to(config.device)
        
        """Cuts off the end of the data such that the length of audio and labels match exactly"""
        end_X = len(X[0,:,0,0])%80
        X = X[:,:,:,0:-end_X]
        
        """Concatenate audio, VAD labels and noise labels"""
        if len(X[0,0,0,:]) != 0:
            concat_X = torch.cat((concat_X, X),3)
            concat_y = torch.cat((concat_y, y),1)
            concat_noise = torch.cat((concat_noise, noise_vector_ini),1)
            
        """The main training loop. Runs after sufficient files are concatenated"""
        if batch > last_batch + concats:
            last_batch = batch # Counter variable
            
            """Stores the data in original variable names and resets the tensors containing the concatenated files"""
            X = concat_X
            y = concat_y
            noise = concat_noise
            
            concat_X = torch.empty((1,1,1,0), device = config.device)
            concat_y = torch.empty((1,0), device = config.device)
            concat_noise = torch.empty((1,0), device = config.device)
            
            """Forward step"""
            pred_DB, pred_AN = config.VAD(X[0,:,:,:].float(), training=1)
    
            loss_DB = calc_loss(y,pred_DB)
            loss_AN = calc_loss_noisetypes(noise[:,0:len(pred_AN[0,0,:])], pred_AN[0,:,:])

            """Variables storing the accumulated losses"""
            total_loss_DB += loss_DB.item() # Accumulated loss over full training epoch
            loss_acc += loss_DB.item()      # Accumulated loss between backward steps
            loss_AN_acc += loss_AN.item()
            comb_loss += (1*loss_DB - config.AN_weight*loss_AN)  # Accumulated combined loss of VAD and noise - including the computational graph
            
            """Calculates the predicted VAD labels and returns the accuracy"""
            accuracy, labs = calc_accuracy(pred_DB.T, y)
            accumulated_accuracy += accuracy
            total_acc += accuracy
            
            """Backward step"""
            if (batch+1) % config.training_batch_size == 0 and batch != 0:
                comb_loss = comb_loss/config.training_batch_size # Finding the mean of the loss
                back_propagation_full(comb_loss, t) # Performs the backpropagation and optimisation step

                after_batches(config.training_batch_size, accumulated_accuracy, loss_acc, batch, loss_DB, loss_AN_acc, X, accuracy, size) # Prints information about the latest forward step to the console. Comment to keep the console clean
                # make_plots(pred_DB.T, X, y, labs) # Plots the waveform, channel scores, predictions and true VAD labels from latest forward step
                
                """Resets variables"""
                loss_acc = 0
                loss_AN_acc = 0
                accumulated_accuracy = 0

                
                
                """ Deletes the accumulated loss including the computational graph"""
                del comb_loss

                """Save information by the end of each training epoch"""
                if batch > config.files_per_epoch:
                    config.training_results_big["training"].append(total_acc/4620)
                    config.training_results_big["learning_rate"].append(config.learning_rate)
                    config.training_results_big["epochs"].append(t)
                    config.training_results_big["loss_DB"].append(total_loss_DB/4620)
                    config.training_results_big["loss_AN"].append(total_loss_AN/4620)
                    return