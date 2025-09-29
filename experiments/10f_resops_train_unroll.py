### TRAIN MANUAL UNROLL LSTM MODELS (1-4) FOR SELECTED RESOPS RESERVOIRS INDIVIDUALLY, SAVE R2 SCORES ###

# Workaround: add directory of 'src' and 'ssjrb_wrapper' to the sys.path
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..')) # One level up to the project root
sys.path.append(parent_dir)

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
from src.data.implied_storage import get_implied_storage
from wakepy import keep

def get_device():
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    
    # Check for CUDA
    elif torch.cuda.is_available():
        return torch.device("cuda")
    
    # Default to CPU
    else:
        return torch.device("cpu")

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

def data_processing(res_id, transform_type, left, right='2020-12-31', train_frac=0.6, val_frac=0.2, test_frac=0.2, return_scaler=False, storage=False):
    """
    Run data processing pipeline for one ResOPS reservoir.
    Params:
    res_id -- int, ResOPS reservoir ID
    transform_type -- str, in preprocessing, whether to 'standardize' or 'normalize' the data
    left -- str (YYYY-MM-DD), beginning boundary of time window
    right -- str (YYYY-MM-DD), end boundary of time window
    return_scaler -- bool, whether or not to return src.data.data_processing.time_scaler() object
    storage -- bool, whether or not to include storage data in features
    """

    # Read in data, columns are [inflow, outflow, storage]
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
    # Add day of the year (doy) as another column
    df['doy'] = df.index.to_series().dt.dayofyear
    # Select data window
    df = df[left:right].copy()

    # Run data processing pipeline
    pipeline = processing_pipeline(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, chunk_size=3*365, pad_value=-1, transform_type=transform_type, fill_na_method='mean')
    # Train/val/test tensors of shape (#chunks, chunksize, [inflow, outflow, storage, doy])
    ts_train, ts_val, ts_test = pipeline.process_data(df) 

    # Separate inputs(X) and targets (y)
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
    
def train_one_reservoir(res_id, left_year, storage, model_num=1, device=torch.device('cpu')):
    """ 
    Train LSTM model for one ResOPS reservoir, return R2 scores for train/val/test
    Params:
    res_id: int, ResOPS reservoir ID
    left_year: int, year corresponding to data left window
    storage: bool, whether or not to include storage in X
    model_num: int, which model architecture to use (1-4)
    """
    # Run data processing pipeline (resulting tuple contains (X, y))
    train_tuple, val_tuple, test_tuple = data_processing(res_id=res_id, transform_type='standardize',
                                                         left=f'{left_year}-01-01', return_scaler=False, storage=storage)

    # Create PyTorch dataset/dataloader for training and validation
    dataset_train, dataset_val = (TensorDataset(*train_tuple), TensorDataset(*val_tuple))
    dataloader_train, dataloader_val = (DataLoader(dataset_train, batch_size=16, shuffle=False), 
                                        DataLoader(dataset_val, batch_size=16, shuffle=False))
    
    # Instantiate model/optimizer using selected archeticture
    if storage:
        input_size = 3 # inflow,  storage, doy
    else:
        input_size = 2 # inflow, doy

    if model_num == 1: # Model 1 (manual unroll)
        hidden_size1 = 30
        hidden_size2 = 15
        output_size = 1 # outflow
        dropout_prob = 0.3
        num_layers = 1

        torch.manual_seed(0)
        model = LSTMModel1(input_size=input_size, hidden_size1=hidden_size1, 
                                    hidden_size2=hidden_size2, output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob)
    
    elif model_num == 2:
        hidden_size1 = 35
        hidden_size2 = 20
        num_layers = 1
        output_size = 1
        dropout_prob = 0.3

        torch.manual_seed(0)
        model = LSTMModel2(input_size=input_size, hidden_size1=hidden_size1, 
                                    hidden_size2=hidden_size2, num_layers=num_layers, output_size=output_size, dropout_prob=dropout_prob, initial_output=0)
    
    elif model_num == 3:
        hidden_size1 = 35
        hidden_size2 = 15
        num_layers = 1
        output_size = 1
        dropout_prob = 0.3

        torch.manual_seed(0)
        model = LSTMModel3(input_size=input_size, hidden_size1=hidden_size1, 
                                    hidden_size2=hidden_size2, output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob, initial_output=0, initial_implied_storage=0)
    
    elif model_num == 4:
        hidden_size = 50
        output_size = 1
        dropout_prob = 0.7
        torch.manual_seed(0)
        model = resRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                    dropout_prob=dropout_prob)
    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Run training loop
    train_losses, val_losses = training_loop(model=model, criterion=criterion, optimizer=optimizer, 
                                            patience=10, dataloader_train=dataloader_train, 
                                            dataloader_val=dataloader_val, epochs=300, device=device)
    
    # Save model
    torch.save(model.state_dict(), f'src/models/saved_models/resops_unroll_models/resops_model{model_num}_{res_id}.pt')

    # Evaluate train/val/test R2 score
    model.to(torch.device('cpu')) # move model back to cpu for eval
    r2_train, r2_val, r2_test = eval_train_val_test(model=model, X_train=train_tuple[0], X_val=val_tuple[0], X_test=test_tuple[0],
                                                    y_train=train_tuple[1], y_val=val_tuple[1], y_test=test_tuple[1])

    return r2_train, r2_val, r2_test

def train_all_reservoirs(res_list, left_years_dict, storage=False, model_num=1, device=torch.device('cpu')):
    """ 
    Train LSTM Model (manual LSTM unroll) for all selected ResOPS reservoirs.
    Params: 
    res_list -- list, list of ResOPS ID's to train models for
    left_years_dict -- dict, maps reservoir ID in res_list to left data window year
    storage -- bool, whether or not to include storage data in X
    model_num -- int, which model architecture to use (1-4)
    """
    # Initialize df to store results
    df = pd.DataFrame(index=res_list, columns=['train', 'val', 'test'])

    for res in res_list:
        # Train models for each reservoir
        r2_train, r2_val, r2_test = train_one_reservoir(res_id=res, left_year=left_years_dict[res], storage=storage, model_num=model_num, device=device)
        # Save results
        df.loc[res, :] = (r2_train, r2_val, r2_test)

    return df

def main():
    # Set the device
    device = get_device()
    print(f"Using device: {device}")

    # Filter reservoirs by record length (80% data record complete)
    res_list = filter_res()

    # Get data window left year for each filtered reservoir
    left_years_dict = get_left_years(res_list=res_list)

    # Train LSTM models for each reservoir
    r2_model1 = train_all_reservoirs(res_list=res_list, left_years_dict=left_years_dict, storage=False, model_num=1, device=device)
    r2_model2 = train_all_reservoirs(res_list=res_list, left_years_dict=left_years_dict, storage=False, model_num=2, device=device)
    r2_model3 = train_all_reservoirs(res_list=res_list, left_years_dict=left_years_dict, storage=False, model_num=3, device=device)
    r2_model4 = train_all_reservoirs(res_list=res_list, left_years_dict=left_years_dict, storage=False, model_num=4, device=device)

    # Save r2 scores
    r2_model1.to_csv('report/results/resops_training/resops_unroll_models/resops_unroll_model1_r2.csv')
    r2_model2.to_csv('report/results/resops_training/resops_unroll_models/resops_unroll_model2_r2.csv')
    r2_model3.to_csv('report/results/resops_training/resops_unroll_models/resops_unroll_model3_r2.csv')
    r2_model4.to_csv('report/results/resops_training/resops_unroll_models/resops_unroll_model4_r2.csv')
    return

# run script
if __name__ == '__main__':
    with keep.running():
        main()