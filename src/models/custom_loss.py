### DEFINES CUSTOM LOSS FUNCTIONS ###

import torch
import torch.nn as nn


class RMSLELoss(nn.Module):
    """
    Implement Root Mean Square Logarithmic Error (RMSLE)
    loss = sqrt(1/n Sum(log(y_hat + 1) - log(y + 1))^2)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))