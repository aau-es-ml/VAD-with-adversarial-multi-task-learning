# VAD-with-adversarial-multi-task-learning
## Description
This repository contains the code used to train a noise-robust VAD using adversarial multi-task learning as presented in [1]. The work is built upon the framework in [2].

The code is split into modules accordingly:

* ```main.py``` The main file of the code. All the other modules are run from this
* ```training.py``` The module responsible for training of the model
* ```testing.py``` The module responsible for validation and testing of the model
* ```dataloaders.py``` The module responsible for loading in the data
* ```file_management.py``` The module responsible for loading and saving the models and results
* ```model_file.py``` The module in which the model is defined
* ```config.py``` The module in which global variables are initialised. From here the learning rate, kernel sizes etc. can be changed.

Additionally, the ground truth VAD labels for the TIMIT database is generated using the .WRD files and given in this repository.

### Dependencies
**Python modules**:
* PyTorch
* pickle
* os
* numpy
* matplitlib.pyplot

The AURORA2 database.
### Executing program
run ```python main.py```

Before executing the program you will have to change the paths to the AURORA2 database in ```config.py```. The VAD labels can be downloaded from https://github.com/zhenghuatan/rVAD
## Citations
[1] C.M. Larsen, P. Koch, Z.-H. Tan. Adversarial Multi-Task Deep Learning for Noise-Robust Voice Activity Detection with Low Algorithmic Delay, _Manuscript_, 2022.

[2] Yu, Cheng & Hung, Kuo-Hsuan & Lin, I-Fan & Fu, Szu-Wei & Tsao, Yu & Hung, Jeih-weih. (2020). Waveform-based Voice Activity Detection Exploiting Fully Convolutional networks with Multi-Branched Encoders, arXiv preprint arXiv:2006.11139, 2020.




