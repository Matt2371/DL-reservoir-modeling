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
from src.models.train_model import *


### TRAIN LSTM MODEL 1A FOR SELECTED RESOPS RESERVOIRS, SAVE R2 METRICS FOR EACH ###

def filter_res(record_frac):
    """
    Filter ResOPS reservoir ID's that have at least some percentage of a complete inflow/outflow/storage record
    Params:
    record_frac -- float, fraction of complete record required
    """
    # Read inflow/outflow/storage for all ResOps reservoirs
    df_inflow = pd.read_csv("data/ResOpsUS/time_series_single_variable_table/DAILY_AV_INFLOW_CUMECS.csv", parse_dates=True, index_col=0, dtype=np.float32)
    df_outflow = pd.read_csv("data/ResOpsUS/time_series_single_variable_table/DAILY_AV_OUTFLOW_CUMECS.csv", parse_dates=True, index_col=0, dtype=np.float32)
    df_storage = pd.read_csv("data/ResOpsUS/time_series_single_variable_table/DAILY_AV_STORAGE_MCM.csv", parse_dates=True, index_col=0, dtype=np.float32)

    # Columns (reservoirs) where more than 90% of record is complete for each variable
    thresh = len(df_inflow) * 0.9
    res_list = list(set(df_inflow.dropna(thresh=thresh, axis=1).columns) 
                    & set(df_outflow.dropna(thresh=thresh, axis=1).columns)
                    & set(df_storage.dropna(thresh=thresh, axis=1).columns))
    
    return res_list

def get_left_years(res_list):
    """ 
    Get left data window years (first record year after leading NA) for each reservoir ID in res_list.
    Return results as dictionary
    Params:
    res_list -- list of ResOPS ID's to fetch left years for
    """
    # For each filtered reservoir, find first year of avail record after leading NA (left year window)
    left_years_dict = {}
    for res in res_list:
        left_years_dict[res] = resops_fetch_data(res_id=res, 
                                                vars=['inflow', 'outflow', 'storage']).isna().idxmin().max().year
    return left_years_dict

def data_processing(res_id, transform_type, left, right='2020-12-31', train_frac=0.6, val_frac=0.2, test_frac=0.2, log_names=[], return_scaler=False, storage=False):
    """
    Run data processing pipeline.
    Params:
    res_id -- int, ResOPS reservoir ID
    transform_type -- str, in preprocessing, whether to 'standardize' or 'normalize' the data
    left -- str (YYYY-MM-DD), beginning boundary of time window
    right -- str (YYYY-MM-DD), end boundary of time window
    log_names -- list of column names (str) to take log of before running rest of pipeline. E.g. ['inflow', 'outflow', 'storage']
    return_scaler -- bool, whether or not to return src.data.data_processing.time_scaler() object
    storage -- bool, whether or not to include storage data in features
    """

    # Read in data, columns are [inflow, outflow, storage]
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
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
    # select inflow, STORAGE, and doy as input features
    if storage:
        X_train, X_val, X_test = ts_train[:, :, [0, 2, 3]], ts_val[:, :, [0, 2, 3]], ts_test[:, :, [0, 2, 3]]
    else:
        X_train, X_val, X_test = ts_train[:, :, [0, 3]], ts_val[:, :, [0, 3]], ts_test[:, :, [0, 3]]
    # select outflow as target feature
    y_train, y_val, y_test = ts_train[:, :, [1]], ts_val[:, :, [1]], ts_test[:, :, [1]]

    if return_scaler:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), pipeline.scaler
    else:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_one_reservoir(res_id, left_year):
    """ 
    Train Model 1a LSTM for one ResOPS reservoir, return R2 scores for train/val/test
    Params:
    res_id: int, ResOPS reservoir ID
    left_year: int, year corresponding to data left window
    """
    # Run data processing pipeline (resulting tuple contains (X, y))
    train_tuple, val_tuple, test_tuple = data_processing(res_id=res_id, transform_type='standardize',
                                                         left=f'{left_year}-01-01', return_scaler=False, storage=False)

    # Create PyTorch dataset/dataloader for training and validation
    dataset_train, dataset_val = (TensorDataset(*train_tuple), TensorDataset(*val_tuple))
    dataloader_train, dataloader_val = (DataLoader(dataset_train, batch_size=1, shuffle=False), 
                                        DataLoader(dataset_val, batch_size=1, shuffle=False))
    
    # Instantiate model/optimizer using Model 1a archeticture
    input_size = 2 # inflow,  doy
    hidden_size1 = 30
    hidden_size2 = 15
    output_size = 1 # outflow
    dropout_prob = 0.3
    num_layers = 1

    torch.manual_seed(0)
    model = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Run training loop
    train_losses, val_losses = training_loop(model=model, criterion=criterion, optimizer=optimizer, 
                                            patience=10, dataloader_train=dataloader_train, 
                                            dataloader_val=dataloader_val, epochs=300)
    
    # Evaluate train/val/test R2 score
    r2_train, r2_val, r2_test = eval_train_val_test(model=model, X_train=train_tuple[0], X_val=val_tuple[0], X_test=test_tuple[0],
                                                    y_train=train_tuple[1], y_val=val_tuple[1], y_test=test_tuple[1])
    return r2_train, r2_val, r2_test

def train_all_reservoirs(res_list, left_years_dict):
    """ 
    Train Model 1a for all selected ResOPS reservoirs.
    Return df of resulting train/val/test R2
    Params: 
    res_list -- list, list of ResOPS ID's to train models for
    left_years_dict -- dict, maps reservoir ID in res_list to left data window year
    """
    # Initialize df to store results
    df = pd.DataFrame(index=res_list, columns=['train', 'val', 'test'])

    for res in res_list:
        # Train models for each reservoir
        r2_train, r2_val, r2_test = train_one_reservoir(res_id=res, left_year=left_years_dict[res])
        # Save results
        df.loc[res, :] = (r2_train, r2_val, r2_test)

    return df

def main():
    # Filter reservoirs by record length (80% data record complete)
    res_list = filter_res(record_frac=0.8)

    # Get data window left year for each filtered reservoir
    left_years_dict = get_left_years(res_list=res_list)

    # Train LSTM for each reservoir, save r2 metrics
    df_r2 = train_all_reservoirs(res_list=res_list, left_years_dict=left_years_dict)

    # Save results
    df_r2.to_csv('report/results/resops_training/resops_individual_r2.csv')

    return


# run script
if __name__ == '__main__':
    main()