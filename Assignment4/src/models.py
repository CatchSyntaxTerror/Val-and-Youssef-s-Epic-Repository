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
    

    