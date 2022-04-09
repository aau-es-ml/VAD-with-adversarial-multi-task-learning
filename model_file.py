"""File for defining the model, including the number of layers, activation functions, kernel sizes and strides"""

import torch
from torch import nn
import config
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VAD_model(nn.Module):
    def __init__(self):
        """Initialisation of the model"""
        super(VAD_model, self).__init__()
        self.EB1 = nn.Conv1d(1,30,config.k_EB1,stride=1,padding='same')
        torch.nn.init.xavier_normal_(self.EB1.weight)
        torch.nn.init.kaiming_uniform_(self.EB1.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.drop1 = nn.Dropout(p=0.2)
        
        self.EB2 = nn.Conv1d(30,15,config.k_EB2,stride=1,padding='same')
        torch.nn.init.kaiming_uniform_(self.EB2.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.drop2 = nn.Dropout(p=0.2)
        
        self.EB3 = nn.Conv1d(15,7,config.k_EB3,stride=1,padding='same')
        torch.nn.init.kaiming_uniform_(self.EB3.weight, a=0.01, mode='fan_in',nonlinearity='leaky_relu')
        self.relu3 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.drop3 = nn.Dropout(p=0.2)
        
        self.EB4 = nn.Conv1d(7,2,config.k_EB4,stride=1,padding='same')
        self.EB4.weight.data.fill_(0.0001)
        self.relu4 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.drop4 = nn.Dropout(p=0.2)
        
        self.FB = nn.Conv1d(2,2,config.k_FB,stride=80, padding = 'valid')
        torch.nn.init.xavier_normal_(self.FB.weight)
        self.sigmoid1 = nn.Sigmoid()
        self.drop5 = nn.Dropout(p=0.0)
        
        self.DB1 = nn.Conv1d(2,2,config.k_DB1,stride=1,padding='same', groups=2)
        torch.nn.init.xavier_normal_(self.DB1.weight)
        self.sigmoid2 = nn.Sigmoid()
        self.drop6 = nn.Dropout(p=0.2)
        
        self.DB2 = nn.Conv1d(2,2,config.k_DB2,stride=1,padding='same', groups=2)
        torch.nn.init.xavier_normal_(self.DB2.weight)
        self.sigmoid3 = nn.Sigmoid()
        self.drop7 = nn.Dropout(p=0.2)
        
        self.DB3 = nn.Conv1d(2,2,config.k_DB3,stride=1,padding='same', groups=2)
        torch.nn.init.xavier_normal_(self.DB3.weight)
        self.sigmoid4 = nn.Sigmoid()
        self.drop8 = nn.Dropout(p=0.0)
        
        self.DN1 = nn.Conv1d(2,2,config.k_DN1,stride=1,padding='same', groups=1)
        torch.nn.init.xavier_normal_(self.DN1.weight)
        self.sigmoidDN1 = nn.Sigmoid()
        self.drop6 = nn.Dropout(p=0.2)
        
        self.DN2 = nn.Conv1d(2,2,config.k_DN2,stride=1,padding='same', groups=1)
        torch.nn.init.xavier_normal_(self.DN2.weight)
        self.sigmoidDN2 = nn.Sigmoid()
        self.drop7 = nn.Dropout(p=0.2)
        
        self.DN3 = nn.Conv1d(2,5,config.k_DN3,stride=1,padding='same', groups=1)
        torch.nn.init.xavier_normal_(self.DN3.weight)
        self.sigmoidDN3 = nn.Softmax(dim=1)
        self.drop8 = nn.Dropout(p=0.0)
        
    
    def forward(self, x, training = 0):
        """Define the dataflow in the forward step of the model"""
        x = x.to(device)
        x = x/2**15

        x = self.EB1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.EB2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.EB3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.EB4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        
        x=torch.nn.functional.normalize(x,dim=2) # Normalise before the sigmoid layers
        
        x = self.FB(x)
        x = self.sigmoid1(x)

        
        DB = x
        DB = self.DB1(x)
        DB = self.sigmoid2(DB)
        DB = self.drop6(DB)

        DB = self.DB2(DB)
        DB = self.sigmoid3(DB)
        DB = self.drop7(DB)

        DB = self.DB3(DB)
        DB = self.sigmoid4(DB)
        
        DN = self.DN1(x)
        DN = self.sigmoidDN1(DN)
        DN = self.drop6(DN)

        DN = self.DN2(DN)
        DN = self.sigmoidDN2(DN)
        DN = self.drop7(DN)

        DN = self.DN3(DN)
        DN = self.sigmoidDN3(DN)

        return DB, DN


