import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#### COLLECTION OF PYTORCH MODEL CLASSES ####

def lstm_unroll(lstm_cell, hidden_size, input):
    """
    Unroll a lstm cell manually so we have access to cell states.
    Params:
    lstm_cell -- nn.LSTMCell(input size, hidden size) object
    hidden_size -- int, number of hidden states, must match how lstm_cell was instantiated
    input -- tensor of shape (batch, timesteps, input size)
    Returns:
    hidden_states -- sequence of hidden states, (batch, timesteps, hidden size)
    cell_states -- sequence of cell states, (batch, timesteps, hidden size)
    """
    # get dimensions
    N, L, Hin = input.shape # (batch size, timesteps, input size)
    # initialize hidden, cell states
    hx, cx = torch.zeros(N, hidden_size), torch.zeros(N, hidden_size) # (batch, hidden size)

    output = [] # store sequence of hidden states
    cell_states = [] # store sequence of cell states

    for i in range(L):
        hx, cx = lstm_cell(input[:, i, :], (hx, cx))
        output.append(hx)
        cell_states.append(cx)
    
    # concatenate results along new dimension
    output = torch.stack(output, dim=1) # (batch, timesteps, hidden size)
    cell_states = torch.stack(cell_states, dim=1) # (batch, timesteps, hidden size)

    return output, cell_states


class LSTMModel1(nn.Module):
    """Model 1: 1 layer LSTM + 1 layer NN with dropout. Forward pass returns sequence of hidden AND cell states.
    Uses manual lstm_unroll()"""
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob):
        """
        Params:
        input_size -- # of features in input
        hidden_size1 -- # of hidden units in LSTM
        hidden_size2 -- # of hidden units in Feedforward
        output_size -- # of features in output
        dropout_prob -- dropout probability for dropout layers
        """
        super(LSTMModel1, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lstm_cell = nn.LSTMCell(input_size, self.hidden_size1)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(hidden_size1, hidden_size2)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # x is (batch size, timesteps, input size)
        out, cell_states = lstm_unroll(self.lstm_cell, self.hidden_size1, x) # (batch size, timesteps, hidden size 1)
        out = self.dropout1(out)
        out = self.linear1(out) # (batch size, timesteps, hidden size 2)
        out = self.relu1(out)
        out = self.dropout2(out)
        out = self.linear2(out) # (batch size, timesteps, output size)
        return out, cell_states

class LSTMModel1_opt(nn.Module):
    """Same architecture as Model 1, but only returns sequence of hidden states and LAST hidden/cell state.
    Uses efficient PyTorch LSTM() layer"""
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, num_layers=1, dropout_prob=0.5):
        super(LSTMModel1_opt, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(hidden_size1, hidden_size2)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # x is (batch size, timesteps, input size)
        out, (h_n, c_n) = self.lstm(x) # out is (batch size, timesteps, hidden size 1)
        out = self.dropout1(out)
        out = self.linear1(out) # (batch size, timesteps, hidden size 2)
        out = self.relu1(out)
        out = self.dropout2(out)
        out = self.linear2(out) # (batch size, timesteps, output size)
        return out, (h_n, c_n)