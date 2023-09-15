#### CONDUCT GRID SEARCH TO TUNE HYPERPARAMETERS OF -- MODEL 1 (a)  -- ON SHASTA DATA (1944-2022) ####
#### DATA PROCESSING: STANDARDIZED, FILL NAN WITH TRAINING MEAN
#### LOSS FUNCTION: MSE LOSS
#### INPUTS: INFLOW, DOY
#### ALSO TRAINS AND SAVES OPTIMAL MODEL ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_processing import *
from src.data.data_fetching import *
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import *
from src.models.hyperparameter_tuning import *

## DATA PROCESSING ##

def data_processing_1a(name='Shasta', transform_type='standardize'):
    """
    Run data processing pipeline. Return Dataloader objects for training and validation data
    Params:
    name -- str, name of reservoir to read
    transform_type -- str, in preprocessing, whether to 'standardize' or 'normalize' the data
    """

    # Read in data, columns are [inflow, outflow, storage]
    df = usbr_fetch_data(name=name, vars=['inflow', 'outflow', 'storage'])
    # Add day of the year (doy) as another column
    df['doy'] = df.index.to_series().dt.dayofyear
    # Select data window (beginning of 1944 to end of 2022)
    df = df['1944-01-01':'2022-12-31'].copy()

    # Run data processing pipeline
    pipeline = processing_pipeline(train_frac=0.6, val_frac=0.2, test_frac=0.2, chunk_size=3*365, pad_value=-1, transform_type=transform_type, fill_na_method='mean')
    # Train/val/test tensors of shape (#chunks, chunksize, [inflow, outflow, storage, doy])
    ts_train, ts_val, ts_test = pipeline.process_data(df) 

    # Separate inputs(X) and targets (y)
    # select inflow and doy as input features
    X_train, X_val, X_test = ts_train[:, :, [0, 3]], ts_val[:, :, [0, 3]], ts_test[:, :, [0, 3]]
    # select outflow as target feature
    y_train, y_val, y_test = ts_train[:, :, [1]], ts_val[:, :, [1]], ts_test[:, :, [1]]

    # Create PyTorch dataset and dataloader
    dataset_train, dataset_val = (TensorDataset(X_train, y_train), TensorDataset(X_val, y_val))
    # shuffle = False to preserve time order
    dataloader_train, dataloader_val = (DataLoader(dataset_train, batch_size=1, shuffle=False), 
                                                        DataLoader(dataset_val, batch_size=1, shuffle=False))
    
    return dataloader_train, dataloader_val


## CONDUCT GRID SEARCH ##

def grid_search_1a(dataloader_train, dataloader_val):
    """Conduct grid search. Save results as csv"""

    # Define hyperparameter space
    names = ['num_layers', 'hidden1', 'hidden2', 'dropout', 'random_seed']
    arrays = [[1, 2], [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 
            [0.3, 0.5, 0.7], [0, 10, 100, 1000, 10000]]
    grid = exhaustive_grid(arrays=arrays, names=names) # dataframe of shape (#runs, 5 (# params))
    results = grid.copy() # dataframe to save results
    results['epochs_trained'] = np.zeros(grid.shape[0])
    results['val_error'] = np.zeros(grid.shape[0])

    # Loop over grid
    for i in range(grid.shape[0]):
        # Select row of parameters
        params_i = grid.iloc[i, :]
        num_layers = int(params_i.num_layers)
        hidden_size1 = int(params_i.hidden1)
        hidden_size2 = int(params_i.hidden2)
        dropout_prob = params_i.dropout
        random_seed = params_i.random_seed

        input_size = 2 # [inflow, doy]
        output_size = 1 # [outflow]

        # Instantiate model
        torch.manual_seed(random_seed)
        model1a_tune = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model1a_tune.parameters(), lr=0.001)

        # Run training loop and get validation error
        train_losses, val_losses = training_loop(model=model1a_tune, criterion=criterion, optimizer=optimizer, 
                                                patience=10, dataloader_train=dataloader_train, dataloader_val=dataloader_val, epochs=300)
        
        # Update results
        results['epochs_trained'].iloc[i] = len(val_losses)
        results['val_error'].iloc[i] = val_losses[-1]

    # Save results
    results.to_csv('report/results/hyperparameter_tuning/model1a_tuning.csv')
    return


## TRAIN AND SAVE MODEL WITH OPTIMAL HYPERPARAMETERS ##

def train_optimal_model_1a(dataloader_train, dataloader_val):
    """Trains and saves optimal model based on results of grid search"""

    ## Train model with optimal hyperparameters and save
    # Find optimal hyperparameters
    # Load in results from grid search
    grid_df = pd.read_csv('report/results/hyperparameter_tuning/model1a_tuning.csv', index_col=0)
    # Average performance over the random seeds
    num_random_seeds = 5
    grid_df['param_id'] = np.repeat(np.arange(int(len(grid_df) / num_random_seeds)), num_random_seeds)
    grid_df_mean = grid_df.groupby('param_id').mean()
    grid_df_mean.drop(columns=['random_seed'], inplace=True)
    # Save sorted df
    grid_df_mean.sort_values(by=['val_error'], axis=0, inplace=True)
    grid_df_mean.to_csv('report/results/hyperparameter_tuning/model1a_avg_tuning.csv')

    # Instantiate optiamal model
    input_size = 2
    hidden_size1 = int(grid_df_mean.iloc[0].hidden1)
    hidden_size2 = int(grid_df_mean.iloc[0].hidden2)
    output_size = 1
    dropout_prob = grid_df_mean.iloc[0].dropout
    num_layers = int(grid_df_mean.iloc[0].num_layers)

    torch.manual_seed(0)
    model1a = LSTMModel1(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model1a.parameters(), lr=0.001)

    # Run training loop
    train_losses, val_losses = training_loop(model=model1a, criterion=criterion, optimizer=optimizer, 
                                            patience=10, dataloader_train=dataloader_train, 
                                            dataloader_val=dataloader_val, epochs=300)
    # Plot train/validation plot
    plot_train_val(train_losses=train_losses, val_losses=val_losses)
    plt.savefig('jobs/model1a_training.png')

    # Save model
    torch.save(model1a.state_dict(), 'src/models/saved_models/model1a.pt')
    return

# run script
if __name__ == '__main__':
    # data processing
    dataloader_train, dataloader_val = data_processing_1a(name='Shasta', transform_type='standardize')

    # # grid search
    # grid_search_1a(dataloader_train=dataloader_train, dataloader_val=dataloader_val)

    # save optimal model
    train_optimal_model_1a(dataloader_train=dataloader_train, dataloader_val=dataloader_val)
