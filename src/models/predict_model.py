import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#### FUNCTIONS TO MAKE PREDICTIONS WITH TRAINED MODELS ####

def predict(model, x):
    """
    Make predictions from a trained model.
    Params:
    model -- trained PyTorch model
    x -- input features
    Returns:
    output -- model output on x
    """

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output, _ = model(x)
    
    return output


def flatten_rm_pad(y_hat, y, pad_value=-1):
    """ 
    Preprocessing step for model evaluation: flatten model output tensors and remove 
    previously padded values on tail
    Parmas:
    y_hat -- tensor of shape (# batches, timesteps, 1), model output
    y -- tensor of shape (# batches, timesteps, 1), true labels
    pad_value -- int, padding value used previously on the true labels
    Returns:
    (y_hat_rm, y_rm) -- tensors of shape (total timesteps, ) with trailing pad indices removed
    """
    assert y_hat.shape == y.shape # Sanity check

    # Flatten arrays
    y_hat = y_hat.flatten()
    y = y.flatten()

    # Remove pad values from true labels
    y_rm = y[y != pad_value]
    # Sanity check for intermediate padded values
    assert sum((y == pad_value)[:len(y_rm)]) == 0

    # Remove pad values from predicted labels
    y_hat_rm = y_hat[:len(y_rm)]

    assert y_hat_rm.shape == y_rm.shape # Sanity check
    return y_hat_rm, y_rm

def plot_predicted_true(y_hat, y, alpha=1, ax=None):
    """
    Plot predicted vs. true values
    Params: 
    y_hat, y -- tensors of shape (timesteps, ), predicted and true sequences
    alpha -- transparency
    ax -- matplotlib axes, plot if provided
    """

    if ax is not None:
        ax.plot(y, label='observed', alpha=alpha)
        ax.plot(y_hat, label='predicted', alpha=alpha)
        ax.set_xlabel('Time')
        ax.set_ylabel('Scaled releases')
        ax.legend()
    else:
        plt.clf()
        plt.plot(y, label='observed', alpha=alpha)
        plt.plot(y_hat, label='predicted', alpha=alpha)
        plt.xlabel('Time')
        plt.ylabel('Scaled releases')
        plt.legend()

    


