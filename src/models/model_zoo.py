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
        hx, cx = lstm_cell(input[:, i, :], (hx, cx)) # get input for current timestep and run through lstm cell
        output.append(hx)
        cell_states.append(cx)
    
    # concatenate results along new dimension
    output = torch.stack(output, dim=1) # (batch, timesteps, hidden size)
    cell_states = torch.stack(cell_states, dim=1) # (batch, timesteps, hidden size)

    return output, cell_states


### FIXME: DEVELOP SUPPORT FOR MULTILAYER LSTM
class LSTMModel1(nn.Module):
    """Model 1: n layer LSTM + 1 layer NN with dropout. Forward pass returns sequence of hidden AND cell states.
    Uses manual lstm_unroll()"""
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, num_layers, dropout_prob):
        """
        Params:
        input_size -- # of features in input
        hidden_size1 -- # of hidden units in LSTM
        hidden_size2 -- # of hidden units in Feedforward
        output_size -- # of features in output
        num_layers -- # of LSTM layers
        dropout_prob -- dropout probability for dropout layers
        """
        super(LSTMModel1, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        # create list of LSTMCell objects
        # the first LSTMCell takes input size and outputs hidden_size1, the following LSTMCells takes hidden_size1 and outputs hidden_size1
        temp_list = [nn.LSTMCell(input_size, self.hidden_size1)] + [nn.LSTMCell(self.hidden_size1, self.hidden_size1) for _ in range(self.num_layers - 1)]
        self.lstm_cell_list = nn.ModuleList(temp_list)

        # define dropout and feedforward layers
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(hidden_size1, hidden_size2)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        """
        Run the forward pass.
        Params:
        x -- model input of shape (batch size, timesteps, input_size)
        Returns:
        out -- model output of shape (batch size, timesteps, output_size)
        (hidden_state_list, cell_state_list) -- lists of model hidden and cell states for each layer, each of shape (batch size, timestpes, hidden_size)
        """
        # initialize lists to store hidden and cell states tensors, each of shape (batch size, timesteps, hidden size 1), for each layer
        hidden_state_list = []
        cell_state_list = []

        # initialize input into first LSTM layer
        input = x # x is (batch size, timesteps, input size)

        # unroll layers one at a time
        for i, lstm_layer in enumerate(self.lstm_cell_list):
            # get hidden and cell states for layer i, both of shape (batch size, timesteps, hidden size 1)
            lstm_hidden_i, lstm_cell_i = lstm_unroll(lstm_cell=lstm_layer, hidden_size=self.hidden_size1, input=input)

            # save results to hidden and cell states list for layer i
            hidden_state_list.append(lstm_hidden_i)
            cell_state_list.append(lstm_cell_i)

            # update input for next iteration (for higher lstm layers, inputs are the hidden states of the previous lstm layer)
            input = lstm_hidden_i # (batch size, timesteps, hidden size 1)

        # continue forward pass with hidden states from the last lstm layer
        out = hidden_state_list[-1] # (batch size, timesteps, hidden size 1)
        # feed-forward
        out = self.dropout1(out)
        out = self.linear1(out) # (batch size, timesteps, hidden size 2)
        out = self.relu1(out)
        out = self.dropout2(out)
        out = self.linear2(out) # (batch size, timesteps, output size)
        return out, (hidden_state_list, cell_state_list)

class LSTMModel1_opt(nn.Module):
    """Same architecture as Model 1, but only returns sequence of hidden states and LAST hidden/cell state.
    Uses efficient PyTorch LSTM() layer"""
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, num_layers, dropout_prob):
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

### FIXME: DEVELOP SUPPORT FOR MULTILAYER LSTM
class LSTMModel2(nn.Module):
    """
    Model 2: (autoregressive) 1 layer LSTM + 1 layer NN with dropout - predictions from previous timestep are
    fed back in as input for the next timestep. Forward pass returns sequence of outputs AND cell states.
    (uses manual lstm_unroll())
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob, initial_output=0):
        """
        Params:
        input_size -- # of features in input (EXCLUDING autoregressive input)
        hidden_size1 -- # of hidden units in LSTM
        hidden_size2 -- # of hidden units in Feedforward
        output_size -- # of features in output
        dropout_prob -- dropout probability for dropout layers
        initial_output -- initialize "previous" output for first timestep
        """
        super(LSTMModel2, self).__init__()
        self.inital_output = initial_output
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lstm_cell = nn.LSTMCell(input_size + 1, self.hidden_size1) # + 1 to account for prev output
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(hidden_size1, hidden_size2)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # get dimensions
        N, L, Hin = x.shape # input shape is (batch size, timesteps, input size)
        # initialize hidden, cell states
        hx, cx = torch.zeros(N, self.hidden_size1), torch.zeros(N, self.hidden_size1) # (batch, hidden size)

        hidden_states = [] # store sequence of hidden states
        cell_states = [] # store sequence of cell states
        output = [] # store predictions
        prev_output = self.inital_output * torch.ones((N, 1)) # initialize previous output (batch size, output size)

        for i in range(L):
            # append previous output to input of current timestep: (batch size, input size) -> (batch size, input size + 1)
            input_i = torch.cat([x[:, i, :], prev_output], dim=-1)
            hx, cx = self.lstm_cell(input_i, (hx, cx)) # hx, cx is of shape (batch size, hidden size 1)

            # save hidden and cell states
            hidden_states.append(hx)
            cell_states.append(cx)

            # add feedforward layers to get output
            out = self.dropout1(hx) # (batch size, hidden size 1)
            out = self.linear1(out) # (batch size, hidden size 2)
            out = self.relu1(out)
            out = self.dropout2(out)
            out = self.linear2(out) # (batch size, output size)

            # save results
            output.append(out)
            prev_output = out
    
        # concatenate results along new dimension (create timesteps dimension at dim=1)
        hidden_states = torch.stack(hidden_states, dim=1) # (batch, timesteps, hidden size1)
        cell_states = torch.stack(cell_states, dim=1) # (batch, timesteps, hidden size1)
        output = torch.stack(output, dim=1) # (batch, timesteps, output size)

        return output, cell_states

### FIXME: DEVELOP SUPPORT FOR MULTILAYER LSTM
class LSTMModel3(nn.Module):
    """
    Model 3: 1 layer LSTM + 1 layer NN with dropout - also inputs inplied storage (prev implied storage + current inflow - prev predicted outflow).
    Forward pass returns sequence of hidden AND implied storages (uses manual lstm_unroll())
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob, initial_output=0, initial_implied_storage=0):
        """
        Params:
        input_size -- # of features in input (EXCLUDING implied storage)
        hidden_size1 -- # of hidden units in LSTM
        hidden_size2 -- # of hidden units in Feedforward
        output_size -- # of features in output
        dropout_prob -- dropout probability for dropout layers
        initial_output -- initialize "previous" output for first timestep
        initial_implied_storage -- initialize "previous" implied storage for first timestep
        """
        super(LSTMModel3, self).__init__()
        self.inital_output = initial_output
        self.initial_implied_storage = initial_implied_storage
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lstm_cell = nn.LSTMCell(input_size + 1, self.hidden_size1) # + 1 to account for prev output
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(hidden_size1, hidden_size2)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # get dimensions
        N, L, Hin = x.shape # input shape is (batch size, timesteps, input size)
        # initialize hidden, cell states
        hx, cx = torch.zeros(N, self.hidden_size1), torch.zeros(N, self.hidden_size1) # (batch, hidden size)

        hidden_states = [] # store sequence of hidden states
        cell_states = [] # store sequence of cell states
        implied_storages = [] # store sequence of implied storages
        output = [] # store predictions
        prev_output = self.inital_output * torch.ones((N, 1)) # initialize previous output (batch size, output size=1)
        prev_implied_storage = self.initial_implied_storage * torch.ones((N, 1)) # initialize previous implied storage (batch size, output size=1)

        for i in range(L):
            # calculate current implied storage: prev storage + current inflow - prev output
            current_implied_storage = prev_implied_storage + x[:, i, [0]] - prev_output
            # append previous implied storage to input of current timestep: (batch size, input size) -> (batch size, input size + 1)
            input_i = torch.cat([x[:, i, :], current_implied_storage], dim=-1)
            hx, cx = self.lstm_cell(input_i, (hx, cx)) # hx, cx is of shape (batch size, hidden size 1)

            # save hidden and cell states, implied storage
            hidden_states.append(hx)
            cell_states.append(cx)
            implied_storages.append(current_implied_storage)

            # add feedforward layers to get output
            out = self.dropout1(hx) # (batch size, hidden size 1)
            out = self.linear1(out) # (batch size, hidden size 2)
            out = self.relu1(out)
            out = self.dropout2(out)
            out = self.linear2(out) # (batch size, output size)

            # save results, update prev_output and prev_implied_storage for next timestep
            output.append(out)
            prev_output = out
            prev_implied_storage = current_implied_storage
    
        # concatenate results along new dimension (create timesteps dimension at dim=1)
        hidden_states = torch.stack(hidden_states, dim=1) # (batch, timesteps, hidden size1)
        cell_states = torch.stack(cell_states, dim=1) # (batch, timesteps, hidden size1)
        implied_storages = torch.stack(implied_storages, dim=1) # (batch, timesteps, output size = 1)
        output = torch.stack(output, dim=1) # (batch, timesteps, output size = 1)

        return output, implied_storages