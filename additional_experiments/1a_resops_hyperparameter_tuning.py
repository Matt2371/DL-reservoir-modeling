#### CONDUCT GRID SEARCH TO TUNE HYPERPARAMETERS ON RESOPS DATA ####
#### DATA PROCESSING: STANDARDIZED, FILL NAN WITH TRAINING MEAN
#### LOSS FUNCTION: MSE LOSS
#### INPUTS: INFLOW, DOY
#### ALSO TRAINS AND SAVES OPTIMAL MODEL ####

# Workaround: add directory of 'src' and 'ssjrb_wrapper' to the sys.path
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..')) # One level up to the project root
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from math import floor
import copy

from src.data.data_processing import *
from src.data.data_fetching import *
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import *
from src.models.hyperparameter_tuning import *

def data_processing(res_id, transform_type, left, right='2020-12-31', train_frac=0.6, val_frac=0.2, test_frac=0.2, log_names=[], return_scaler=False, storage=False):
    """
    Run data processing pipeline for one ResOPS reservoir.
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

def train_one_config(res_id, left_year, storage, num_layers, hidden_size1, hidden_size2, dropout_prob, seed, device):
    """ 
    Train Model 1a LSTM for one ResOPS reservoir, for one hyperparameter config, one seed, and return train/val losses
    Params:
    res_id: int, ResOPS reservoir ID
    left_year: int, year corresponding to data left window
    storage: bool, whether or not to include storage in X
    * hyperparemeters
    """
    # Run data processing pipeline (resulting tuple contains (X, y))
    train_tuple, val_tuple, test_tuple = data_processing(res_id=res_id, transform_type='standardize',
                                                         left=f'{left_year}-01-01', return_scaler=False, storage=storage)

    # Create PyTorch dataset/dataloader for training and validation
    dataset_train, dataset_val = (TensorDataset(*train_tuple), TensorDataset(*val_tuple))
    dataloader_train, dataloader_val = (DataLoader(dataset_train, batch_size=1, shuffle=False), 
                                        DataLoader(dataset_val, batch_size=1, shuffle=False))
    
    # Instantiate model/optimizer using Model 1a archeticture
    if storage:
        input_size = 3 # inflow,  storage, doy
    else:
        input_size = 2 # inflow, doy
    output_size = 1 # outflow

    torch.manual_seed(seed)
    model = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Run training loop
    train_losses, val_losses = training_loop(model=model, criterion=criterion, optimizer=optimizer, 
                                            patience=10, dataloader_train=dataloader_train, 
                                            dataloader_val=dataloader_val, epochs=300, device=device)
    # # Save model
    # torch.save(model.state_dict(), f'src/models/saved_models/resops_model1/resops_model{"1S" if storage else "1"}_{res_id}.pt')
    
    # # Evaluate train/val/test R2 score
    # r2_train, r2_val, r2_test = eval_train_val_test(model=model, X_train=train_tuple[0], X_val=val_tuple[0], X_test=test_tuple[0],
    #                                                 y_train=train_tuple[1], y_val=val_tuple[1], y_test=test_tuple[1])
    return train_losses, val_losses

def grid_search_one_reservoir(res_id, left_year, device):
    """Conduct grid search. Return results as df"""

    # Define hyperparameter space
    names = ['num_layers', 'hidden1', 'hidden2', 'dropout', 'random_seed']
    arrays = [[1, 2], [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 
            [0.3, 0.5, 0.7], [0, 10, 100, 1000, 10000]]
    grid = exhaustive_grid(arrays=arrays, names=names) # dataframe of shape (#runs, 5 (# params))
    results = grid.copy() # dataframe to save results
    results['epochs_trained'] = np.zeros(grid.shape[0])
    results['val_error'] = np.zeros(grid.shape[0])

    # Loop over grid
    for i in tqdm(range(grid.shape[0]), desc=f'Grid search reservoir {res_id}: '):
        # Select row of parameters
        params_i = grid.iloc[i, :]
        num_layers = int(params_i.num_layers)
        hidden_size1 = int(params_i.hidden1)
        hidden_size2 = int(params_i.hidden2)
        dropout_prob = params_i.dropout
        random_seed = params_i.random_seed

        train_losses, val_losses = train_one_config(res_id=res_id, left_year=left_year, storage=False, num_layers=num_layers, 
                                                    hidden_size1=hidden_size1, hidden_size2=hidden_size2, dropout_prob=dropout_prob, 
                                                    seed=random_seed, device=device)
        
        # Update results
        results.loc[i, 'epochs_trained'] = len(val_losses)
        results.loc[i, 'val_error'] = val_losses[-1]

    return results

def create_parallel_axis(res_id, result):
    """
    Create and save a parallel-axis plot for one reservoir's grid-search results.

    Params:
    res_id -- int or str, ResOPS reservoir ID
    result -- pd.DataFrame, output from grid_search_one_reservoir()
    """
    group_cols = ['num_layers', 'hidden1', 'hidden2', 'dropout']
    plot_data = (
        result
        .groupby(group_cols, as_index=False)
        .mean(numeric_only=True)
        .drop(columns=['random_seed'], errors='ignore')
        .sort_values(by='val_error', axis=0)
    )
    label_map = {
        'num_layers': 'LSTM layers',
        'hidden1': 'LSTM hidden size',
        'hidden2': 'FF hidden size',
        'dropout': 'Dropout rate'
    }
    category_orders = {
        'num_layers': sorted(plot_data['num_layers'].dropna().unique(), reverse=True),
        'hidden1': sorted(plot_data['hidden1'].dropna().unique(), reverse=True),
        'hidden2': sorted(plot_data['hidden2'].dropna().unique(), reverse=True),
        'dropout': sorted(plot_data['dropout'].dropna().unique())
    }

    dimensions = [
        dict(
            label=label_map[col],
            values=plot_data[col],
            categoryorder='array',
            categoryarray=category_orders[col]
        )
        for col in group_cols
    ]

    fig = go.Figure(
        go.Parcats(
            dimensions=dimensions,
            line=dict(
                color=plot_data['val_error'],
                colorscale=px.colors.sequential.Turbo,
                cmid=plot_data['val_error'].median(),
                colorbar=dict(
                    title=dict(text='Val error', font=dict(size=20)),
                    tickfont=dict(size=20)
                )
            )
        )
    )

    fig.update_layout(
        font=dict(size=20),
        margin=dict(t=140, b=40, l=40, r=140),
        width=1400,
        height=900
    )

    output_dir = os.path.join(
        parent_dir,
        'report',
        'results',
        'additional_experiments',
        'hyperparameter_tuning_resops'
    )
    os.makedirs(output_dir, exist_ok=True)

    file_stem = f'{res_id}_grid_search_model1'
    fig.write_html(os.path.join(output_dir, f'{file_stem}.html'))
    fig.write_image(
        os.path.join(output_dir, f'{file_stem}.png'),
        width=1400,
        height=900,
        scale=2
    )
    return fig, plot_data

def main(n=None, random_seed=0):
    # Get training device
    device = get_device()
    print(f'Using device: {device}')

    # Filter reservoirs by record length (80% data record complete)
    res_list = sorted(filter_res())

    # Select n random reservoirs from res_list if n provided
    if n is not None:
        n = min(n, len(res_list))
        rng = np.random.default_rng(random_seed)
        res_list = rng.choice(res_list, size=n, replace=False).tolist()

    # Get data window left year for each filtered reservoir
    left_years_dict = get_left_years(res_list=res_list)

    # Conduct grid search and produce pallel axis plot for each reservoir
    for res_id in res_list:
        result = grid_search_one_reservoir(res_id=res_id, left_year=left_years_dict[res_id], device=device)
        result.to_csv(f'report/results/additional_experiments/hyperparameter_tuning_resops/logs/{res_id}_grid_search.csv')
        create_parallel_axis(res_id=res_id, result=result)
    return

# run script
if __name__ == '__main__':
    main(n=10)
