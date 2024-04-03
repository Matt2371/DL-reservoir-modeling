## TRAIN A MC-LSTM FOR SHASTA ##

# Modifications to data processing: Separated mass (X_m) and auxillary (X_a) inputs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from math import floor
import os
import copy

from src.data.data_processing import *
from src.data.data_fetching import *
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import plot_train_val
from src.models.train_mclstm import *
from src.models.hyperparameter_tuning import *
from src.models import mclstm

def data_processing(name, transform_type, train_frac=0.6, val_frac=0.2, test_frac=0.2, left='1944-01-01', right='2022-12-31', log_names=[], return_scaler=False):
    """
    Run data processing pipeline.
    Params:
    name -- str, name of reservoir to read
    transform_type -- str, in preprocessing, whether to 'standardize' or 'normalize' the data.
    left -- str (YYYY-MM-DD), beginning boundary of time window
    right -- str (YYYY-MM-DD), end boundary of time window
    log_names -- list of column names (str) to take log of before running rest of pipeline. E.g. ['inflow', 'outflow', 'storage']
    return_scaler -- bool, whether or not to return src.data.data_processing.time_scaler() object
    """

    # Read in data, columns are [inflow, outflow, storage]
    df = usbr_fetch_data(name=name, vars=['inflow', 'outflow', 'storage'])
    # Add day of the year (doy) as another column
    df['doy'] = df.index.to_series().dt.dayofyear
    # Select data window
    df = df[left:right].copy()

    # Take log of df columns that are in log_names
    for column_name in df.columns:
        if column_name in log_names:
            df[column_name] = np.log(df[column_name])
        else:
            continue

    # Run data processing pipeline
    pipeline = processing_pipeline(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, chunk_size=3*365, pad_value=-1, transform_type=transform_type, fill_na_method='mean')
    # Train/val/test tensors of shape (#chunks, chunksize, [inflow, outflow, storage, doy])
    ts_train, ts_val, ts_test = pipeline.process_data(df) 

    # Separate inputs(X) and targets (y)
    X_train_a, X_val_a, X_test_a = ts_train[:, :, [3]], ts_val[:, :, [3]], ts_test[:, :, [3]] # auxillary input = doy
    X_train_m, X_val_m, X_test_m = ts_train[:, :, [0]], ts_val[:, :, [0]], ts_test[:, :, [0]] # mass input = inflow

    # select outflow as target feature
    y_train, y_val, y_test = ts_train[:, :, [1]], ts_val[:, :, [1]], ts_test[:, :, [1]]

    if return_scaler:
        return (X_train_m, X_train_a, y_train), (X_val_m, X_val_a, y_val), (X_test_m, X_test_a, y_test), pipeline.scaler
    else:
        return (X_train_m, X_train_a, y_train), (X_val_m, X_val_a, y_val), (X_test_m, X_test_a, y_test)
    

def main():
    ## Data processing
    data_result = data_processing(name='Shasta', transform_type='standardize')
    # Create PyTorch Dataset and Dataloader
    dataset_train_res, dataset_val_res = (TensorDataset(*data_result[0]), 
                                        TensorDataset(*data_result[1]))
    dataloader_train_res, dataloader_val_res = (DataLoader(dataset_train_res, batch_size=1, shuffle=False), 
                                                DataLoader(dataset_val_res, batch_size=1, shuffle=False))

    ## Model training
    # Instantiate model and optimizer
    torch.manual_seed(0)
    criterion = nn.MSELoss()
    mc_lstm = mclstm.MassConservingLSTM(in_dim=1, aux_dim=1, out_dim=30, batch_first=True)
    optimizer = optim.Adam(mc_lstm.parameters(), lr=0.001)
    # Run training loop (finetune model to shasta)
    train_losses, val_losses = training_loop_mclstm(model=mc_lstm, criterion=criterion, optimizer=optimizer, 
                                            patience=10, dataloader_train=dataloader_train_res, 
                                            dataloader_val=dataloader_val_res, epochs=300)
    # Save trained model
    torch.save(mc_lstm.state_dict(), 'src/models/saved_models/mc_lstm.pt')

    # Plot train/validation plot
    plot_train_val(train_losses=train_losses, val_losses=val_losses)
    plt.savefig('jobs/mclstm_training.png')
    return

# run script
if __name__ == '__main__':
    main()
