import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

#### FUNCTIONS TO MAKE PREDICTIONS WITH TRAINED MODELS AND PLOT RESULTS ####

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

def plot_and_eval(model, X_train, X_val, X_test, y_train, y_val, y_test, datetime_index, ax, text_ypos, alpha=1):
    """ 
    Plot true and predicted labels for the entire dataset (train + val + test), as well as calculate and display R2 metrics for each
    Params:
    model -- PyTorch model of interest
    X_train/X_val/X_test -- input data of shape (# batches, timesteps, # features)
    y_train/y_val/y_test -- target data of shape (# batches, timesteps, 1)
    datetime_index -- datetime index for ENTIRE DATASET
    ax -- matplotlib axes to plot on
    text_ypos -- y coordinate to print r2 measure on plot
    alpha -- plotting transparency
    Returns:
    (r2_train, r2_val, r2_test) -- tuple of r2 metrics for the train, val, and test sets
    """
    # Get predictions
    y_hat_train, y_hat_val, y_hat_test = predict(model, X_train), predict(model, X_val), predict(model, X_test)

    # Flatten and remove padding values
    y_hat_train, y_train = flatten_rm_pad(y_hat=y_hat_train, y=y_train)
    y_hat_val, y_val = flatten_rm_pad(y_hat=y_hat_val, y=y_val)
    y_hat_test, y_test = flatten_rm_pad(y_hat=y_hat_test, y=y_test)

    # Get length of training/val/test datasets
    train_len, val_len, test_len = len(y_train), len(y_val), len(y_test)

    # Find R2 metrics
    r2_train = r2_score(y_pred=y_hat_train, y_true=y_train)
    r2_val = r2_score(y_pred=y_hat_val, y_true=y_val)
    r2_test = r2_score(y_pred=y_hat_test, y_true=y_test)
    metrics = (r2_train, r2_val, r2_test)

    # Concat results, check dimensions with datetime index
    y_hat, y = torch.cat((y_hat_train, y_hat_val, y_hat_test)), torch.cat((y_train, y_val, y_test))
    assert len(datetime_index) == len(y_hat) == len(y)

    # Plot
    ax.plot(datetime_index, y, label='observed', alpha=alpha)
    ax.plot(datetime_index, y_hat, label='predicted', alpha=alpha)
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Scaled releases')
    ax.legend()
    # label training/val/test sets and their respective r2
    ax.axvline(x=datetime_index[train_len - 1], linestyle='--', color='black', alpha=alpha)
    ax.text(x=datetime_index[train_len], y=text_ypos, s=f'Training data: $R^2={round(r2_train, 2)}$ ', ha='right', va='top', size='large')

    ax.axvline(x=datetime_index[train_len + val_len - 1], linestyle='--', color='black', alpha=alpha)
    ax.text(x=datetime_index[train_len + val_len - 1], y=text_ypos, s=f'Validation data: $R^2={round(r2_val, 2)}$ ', ha='right', va='top', size='large')

    ax.axvline(x=datetime_index[train_len + val_len + test_len - 1], linestyle='--', color='black', alpha=alpha)
    ax.text(x=datetime_index[train_len + val_len + test_len - 1], y=text_ypos, s=f'Test data: $R^2={round(r2_test, 2)}$ ', ha='right', va='top', size='large')

    return metrics

    


