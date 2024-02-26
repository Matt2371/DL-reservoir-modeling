### FUNCTIONS TO SUPPORT EVALUATING MODELS USING AUTOREGRESSIVE LAGS ###
### USEFUL FOR NON-LSTM BENCHMARK MODELS ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from src.data.data_processing import *
from src.data.data_fetching import *
from src.data.data_processing_lagged import *


def plot_lagged_model(model, X_train, X_val, X_test, y_train, y_val, y_test, datetime_index, ax, text_ypos, alpha=1):
    """ 
    Plot true and predicted labels for the entire dataset (train + val + test), as well as calculate and display R2 metrics for each
    Params:
    model -- sklearn model of interest
    X_train/X_val/X_test -- input data of shape (# timesteps, # features)
    y_train/y_val/y_test -- target data of shape (# timesteps, 1)
    datetime_index -- datetime index for ENTIRE DATASET
    ax -- matplotlib axes to plot on
    text_ypos -- y coordinate to print r2 measure on plot
    alpha -- plotting transparency
    Returns:
    (r2_train, r2_val, r2_test) -- tuple of r2 metrics for the train, val, and test sets
    """
    # Get predictions
    y_hat_train, y_hat_val, y_hat_test = model.predict(X_train), model.predict(X_val), model.predict(X_test)

    # Get length of training/val/test datasets
    train_len, val_len, test_len = len(y_train), len(y_val), len(y_test)

    # Find R2 metrics
    r2_train = r2_score(y_pred=y_hat_train, y_true=y_train)
    r2_val = r2_score(y_pred=y_hat_val, y_true=y_val)
    r2_test = r2_score(y_pred=y_hat_test, y_true=y_test)
    metrics = (r2_train, r2_val, r2_test)

    # Concat results, check dimensions with datetime index
    y_hat, y = np.concatenate((y_hat_train, y_hat_val, y_hat_test)), np.concatenate((y_train, y_val, y_test))
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

def eval_train_val_test_lagged(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """ 
    Evaluate R2 metrics for train/val/test set
    Params:
    model -- sklearn model of interest
    X_train/X_val/X_test -- input data of shape (# timesteps, # features)
    y_train/y_val/y_test -- target data of shape (# timesteps, 1)
    Returns:
    r2_train, r2_val, r2_test -- r2 metrics for the train, val, and test sets
    """
    # Get predictions
    y_hat_train, y_hat_val, y_hat_test = model.predict(X_train), model.predict(X_val), model.predict(X_test)

    # Find R2 metrics
    r2_train = r2_score(y_pred=y_hat_train, y_true=y_train)
    r2_val = r2_score(y_pred=y_hat_val, y_true=y_val)
    r2_test = r2_score(y_pred=y_hat_test, y_true=y_test)

    return r2_train, r2_val, r2_test