"""Main file for training and testing the VAD """

from torch.utils.data import DataLoader
import training
import testing
from dataloaders import AURORA2_test, AURORA2_train
import config
import file_management




def run_train(t):
    print(f"Epoch {t+1}\n--------------TRAIN-----------------")
    config.VAD.train()
    dataset_train = AURORA2_train() # Creates an instance of the dataloader class of the training set
    train_data_loader = DataLoader(dataset_train, batch_size=1, shuffle=True) # The data is randomly shuffled
    training.update_learning_rates()
    training.train_loop(train_data_loader, t) # The main training loop
    
    """Updates the learning rate after each epoch"""
    config.learning_rate *= config.LR_factor
    config.learning_rate_DN *= config.LR_factor
    
def run_validation(t):
    
    """Test of accuracy on validation set. One test for each combination of SNR level and noise type"""
    print(f"Epoch {t+1}\n--------------VALIDATION-----------------")
    noises = ["N1", "N2", "N3", "N4"]
    SNRs = ["-5", "0", "5", "10", "15", "20", "CLEA"]
    config.validation = 1
    config.VAD.eval()
    for j in noises:
        for k in SNRs:
            print(f"Noise type: {j} - SNR level: {k}")
            config.SNR_level_AURORA = k
            config.noise_type_AURORA = j
            dataset_test = AURORA2_test() # Creates an instance of the dataloader class of the validation split
            test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False) # The data is not shuffled and thus the order is kept the same every time
            testing.validation_loop(test_loader, t) # The main validation loop
    
def run_test(t):
    
    """Test on testing set. One test for each combination of SNR level and noise type"""
    noises = ["N1", "N2", "N3", "N4"]
    SNRs = ["-5", "0", "5", "10", "15", "20", "CLEA"]
    config.validation = 1
    config.VAD.eval()
    for j in noises:
        for k in SNRs:
            config.SNR_level_AURORA = k
            config.noise_type_AURORA = j
            dataset_test = AURORA2_test() # Creates an instance of the dataloader class of the testing split
            test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False) # The data is not shuffled and thus the order is kept the same every time
            testing.testing_loop(test_loader, t) # The main testing loop
    file_management.save_results_AUC(config.training_results_AUC, t)
    

def count_parameters(model):
    """Counts the number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    
    # file_management.load_model() # Include this if you wish to load a pretrained model
    # file_management.save_model_initial(config.WVAD_model) # Include this if you wish to save the randomly initiated model
    epochs = config.training_epochs
    
    """ Training loop including test of accuracy on validation set after each epoch"""
    for t in range(epochs):

        run_train(t)
        run_validation(t)
        file_management.save_results(config.training_results_big, t)
        file_management.save_model(config.VAD, t)
    run_test(t)

    print("Done!")
# 