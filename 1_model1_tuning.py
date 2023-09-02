#### Conduct grid search to tune hyperparameters of model 1 on Shasta data (1944-2022) ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

## Data Processing
from src.data.data_processing import *
from src.data.data_fetching import *

# Read in data, columns are [inflow, outflow, storage]
df = usbr_fetch_data(name='Shasta', vars=['inflow', 'outflow', 'storage'])
# Add day of the year (doy) as another column
df['doy'] = df.index.to_series().dt.dayofyear
# Select data window (beginning of 1944 to end of 2022)
df = df['1944-01-01':'2022-12-31'].copy()

# Run data processing pipeline
pipeline = processing_pipeline(train_frac=0.6, val_frac=0.2, test_frac=0.2, chunk_size=3*365, pad_value=-1)
# Train and val of shape (#chunks, chunksize, [inflow, outflow, storage, doy]), test of shape (timesteps, [inflow, outflow, storage, doy])
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


## Conduct Grid Search
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import *
from src.models.hyperparameter_tuning import *

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
    model1_tune = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                             hidden_size2=hidden_size2, output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model1_tune.parameters(), lr=0.001)

    # Run training loop and get validation error
    train_losses, val_losses = training_loop(model=model1_tune, criterion=criterion, optimizer=optimizer, 
                                             patience=10, dataloader_train=dataloader_train, dataloader_val=dataloader_val, epochs=200)
    
    # Update results
    results['epochs_trained'].iloc[i] = len(val_losses)
    results['val_error'].iloc[i] = val_losses[-1]

# Save results
results.to_csv('report/results/hyperparameter_tuning/model1_tuning.csv')