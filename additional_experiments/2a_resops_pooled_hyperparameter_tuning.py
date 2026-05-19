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
import os
import copy
from tqdm import tqdm
import geopandas as gpd

from src.data.data_processing import *
from src.data.data_fetching import *
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import *
from src.models.hyperparameter_tuning import *

from additional_experiments.pooled_training_utils import (get_attributes, 
                                                     data_processing, 
                                                     multi_reservoir_data_oos, 
                                                     train_simultaneous_model)

def grid_search(data_result, device=torch.device("cpu")):
    """
    Conduct grid search for pooled ResOPS model. Return results as df
    data_result -- output of multi_reservoir_data_oos.fetch_and_combine()
    """

    # Define hyperparameter space
    names = ['num_layers', 'hidden1', 'hidden2', 'dropout', 'random_seed']
    arrays = [[1, 2], [20, 40, 60, 80, 100, 120], [20, 40, 60, 80, 100, 120], 
            [0.3, 0.5, 0.7], [0, 10, 100, 1000, 10000]]
    grid = exhaustive_grid(arrays=arrays, names=names) # dataframe of shape (#runs, 5 (# params))
    results = grid.copy() # dataframe to save results
    results['epochs_trained'] = np.zeros(grid.shape[0])
    results['val_error'] = np.zeros(grid.shape[0])

    # Loop over grid
    for i in tqdm(range(grid.shape[0]), desc=f'Grid search pooled ResOPS: '):
        # Select row of parameters
        params_i = grid.iloc[i, :]
        num_layers = int(params_i.num_layers)
        hidden_size1 = int(params_i.hidden1)
        hidden_size2 = int(params_i.hidden2)
        dropout_prob = params_i.dropout
        random_seed = params_i.random_seed

        train_losses, val_losses, model = train_simultaneous_model(X_train=data_result[0][0], y_train=data_result[0][1],
                                            X_val=data_result[1][0], y_val=data_result[1][1], 
                                            hidden_size1=hidden_size1, hidden_size2=hidden_size2, dropout_prob=dropout_prob,
                                            num_layers=num_layers, random_seed=random_seed, device=device)
        
        # Update results
        results.loc[i, 'epochs_trained'] = len(val_losses)
        results.loc[i, 'val_error'] = val_losses[-1]

        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results

def create_parallel_axis(result):
    """
    Create and save a parallel-axis plot for one reservoir's grid-search results.

    Params:
    result -- pd.DataFrame, output from grid_search()
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
        'hyperparameter_tuning_pooled'
    )
    os.makedirs(output_dir, exist_ok=True)

    file_stem = f'grid_search_model1_pooled'
    fig.write_html(os.path.join(output_dir, f'{file_stem}.html'))
    fig.write_image(
        os.path.join(output_dir, f'{file_stem}.png'),
        width=1400,
        height=900,
        scale=2
    )
    return fig, plot_data

def main():
    # Set the device
    device = get_device()
    print(f"Using device: {device}")

    # -----   Data collection and processing    ----- #
    attribute_df = get_attributes(index_type=str)    # Get attributes df
    res_list = filter_res()    # Get list of reservoir ID of interest

    # Randomly choose out of sample reservoirs
    np.random.seed(0)
    oos_list = np.random.choice(a=sorted(res_list, key=int), size=round(0.3 * len(res_list)), replace=False)

    # Get left window years
    left_years_dict = get_left_years(res_list=res_list)

    # Data collection and processing
    data_combiner = multi_reservoir_data_oos(left_years_dict=left_years_dict, res_list=res_list, oos_list=oos_list, attributes=attribute_df)
    data_result = data_combiner.fetch_and_combine()


    # -----   Grid search for hyperparameter tuning   ----- #
    grid_search_df = grid_search(data_result=data_result, device=device)
    # print(grid_search_df.groupby(['num_layers', 'hidden1', 'hidden2', 'dropout'], as_index=False)
    #     .mean(numeric_only=True)
    #     .drop(columns=['random_seed'], errors='ignore')
    #     .sort_values(by='val_error', axis=0, ascending=True).head(10))
    create_parallel_axis(grid_search_df)
    grid_search_df.to_csv(f'report/results/additional_experiments/hyperparameter_tuning_pooled/grid_search_model1_pooled.csv')


    # -----   Train and save final model with best hyperparameters    ----- #
    best_params = (
        grid_search_df
        .groupby(['num_layers', 'hidden1', 'hidden2', 'dropout'], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=['random_seed'], errors='ignore')
        .sort_values(by='val_error', axis=0, ascending=True)
    ).iloc[0, :]
    print(f"Best hyperparameters: {best_params}")
    hidden_size1, hidden_size2, dropout_prob, num_layers = (int(best_params.hidden1), 
                                                            int(best_params.hidden2), 
                                                            float(best_params.dropout), 
                                                            int(best_params.num_layers))
    # Train in-sample reservoirs simultaneously
    _, _, simul_model = train_simultaneous_model(X_train=data_result[0][0], y_train=data_result[0][1],
                                           X_val=data_result[1][0], y_val=data_result[1][1], 
                                           hidden_size1=hidden_size1, hidden_size2=hidden_size2, dropout_prob=dropout_prob, num_layers=num_layers, 
                                           plot=False, device=device)
    
    # Save model
    torch.save(simul_model.state_dict(), 'report/results/additional_experiments/hyperparameter_tuning_pooled/resops_simul_model.pt')
    
    # -----     Evaluate and save in-sample performance     ----- #

    simul_model.to(torch.device("cpu")) # move model to CPU for evaluation
    r2_in_sample_df = pd.DataFrame(index=data_combiner.is_list, columns=['train', 'val'])
    for in_res in tqdm(data_combiner.is_list, desc='Evaluating in-sample performance: '):
        r2_in_sample_df.loc[in_res, :] = [r2_score_tensor(model=simul_model, X=data_combiner.X_train_dict[in_res], 
                                                          y=data_combiner.y_train_dict[in_res]),
                                          r2_score_tensor(model=simul_model, X=data_combiner.X_val_dict[in_res], 
                                                          y=data_combiner.y_val_dict[in_res])]
    r2_in_sample_df.to_csv('report/results/additional_experiments/hyperparameter_tuning_pooled/resops_oos_in_sample_train_val.csv')

    # -----     Evaluate and save out-of-sample performance     ----- #

    r2_out_sample_df = pd.DataFrame(index=data_combiner.oos_list, columns=['test'])
    for out_res in tqdm(data_combiner.oos_list, desc='Evaluating out-of-sample performance: '):
        r2_out_sample_df.loc[out_res, :] = r2_score_tensor(model=simul_model, X=data_combiner.X_test_dict[out_res], y=data_combiner.y_test_dict[out_res])
    r2_out_sample_df.to_csv('report/results/additional_experiments/hyperparameter_tuning_pooled/resops_oos_out_of_sample_test.csv')

    return

# run script
if __name__ == '__main__':
    main()
    


