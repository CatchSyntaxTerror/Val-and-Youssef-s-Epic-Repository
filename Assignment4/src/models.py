import torch
import torch.nn as nn
"""
Basline RNN, GRU and LSTM models
"""

class RNN(nn.Module):
    """ The BaseLine RNN modified from the textbook to take in the number of layers """
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = hidden[-1,:,:]
        out = self.fc(out)
        return out
    
class LSTM(nn.Module):
    """Taken and adapted from the textbook"""
    def __init__(self,rnn_hidden_size, fc_hidden_size,num_layers):
        super().__init__
        self.rnn = nn.LSTM(1,rnn_hidden_size,num_layers,batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size,fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, cell) = self.rnn(x)
        out = hidden[-1,:,:]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    