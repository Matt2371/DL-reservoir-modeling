import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import unittest
from src.models.model_zoo import *


class test_forward(unittest.TestCase):
    """Test forward passes for models in model zoo"""

    def test_lstm_unroll(self):
        """Test shape of lstm unroll function"""
        class dummy_lstm_unroll(nn.Module):
            def __init__(self):
                super(dummy_lstm_unroll, self).__init__()
                self.lstm_cell = nn.LSTMCell(input_size=2, hidden_size=3) #input size=2, output size=3
                return
            def forward(self, x):
                out, cell_states = lstm_unroll(self.lstm_cell, hidden_size=3, input=x)
                return out, cell_states
        
        dummy_model = dummy_lstm_unroll()
        input = torch.tensor(np.ones((5, 7, 2)), dtype=torch.float) # (batch size, timesteps, input size)
        self.assertEqual(tuple(dummy_model(input)[0].shape), (5, 7, 3)) # expected output shape is (batch size, timesteps, output size)
        return