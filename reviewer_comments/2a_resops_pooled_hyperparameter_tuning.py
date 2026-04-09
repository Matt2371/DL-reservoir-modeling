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


def get_attributes(index_type=str):
    """
    Get reservoir attributes df (categorical attributes one-hot encoded; continuous attributes kept as floats)
    Params:
    index_type -- type of index for returned dataframe, either str or int
    """
    # GRanD Attributes
    gdf = gpd.read_file("data/GRanD/GRanD_dams_v1_3.shp")
    gdf = gdf.drop(columns="geometry").set_index("GRAND_ID")

    # Main reservoir use
    use_ohe = pd.get_dummies(gdf['MAIN_USE'], prefix='USE', dtype='float')
    use_ohe.index = use_ohe.index.astype(str)

    # Capacity (standardize once across reservoirs; append later after time-series preprocessing)
    capacity = gdf['CAP_MCM'].copy()
    capacity.index = capacity.index.astype(str)
    capacity = capacity.fillna(capacity.mean()) # impute missing values with mean
    capacity = (capacity - capacity.mean()) / capacity.std() # standardize

    # Operating agency (one-hot encode)
    operating_agency = pd.read_csv("data/ResOpsUS/attributes/reservoir_attributes.csv", index_col=0)["AGENCY_CODE"]
    operating_agency_ohe = pd.get_dummies(operating_agency, prefix='AGENCY', dtype='float')
    operating_agency_ohe.index = operating_agency_ohe.index.astype(str)

    # DOR category (based on log(mean inflow / max storage))
    df_inflow = pd.read_csv("data/ResOpsUS/time_series_single_variable_table/DAILY_AV_INFLOW_CUMECS.csv", 
                            parse_dates=True, index_col=0, dtype=np.float32)
    df_storage = pd.read_csv("data/ResOpsUS/time_series_single_variable_table/DAILY_AV_STORAGE_MCM.csv", 
                            parse_dates=True, index_col=0, dtype=np.float32)
    df_result = pd.concat([df_inflow.mean(skipna=True), 
                           df_storage.max()], axis=1, join='inner')
    df_result.columns = ['mean_inflow', 'max_storage']

    df_result['log_mean_inflow_max_storage'] = np.log(df_result['mean_inflow'] / df_result['max_storage'])
    df_result['log_mean_inflow_max_storage_cat'] = pd.cut(df_result['log_mean_inflow_max_storage'], bins=[-np.inf,-3.79, -3.17, -2.46, np.inf], labels=['very_high', 'high', 'medium', 'low'])
    dor_ohe = pd.get_dummies(df_result['log_mean_inflow_max_storage_cat'], prefix='DOR', dtype='float')
    dor_ohe.index = dor_ohe.index.astype(str)

    attribute_df = use_ohe.join([dor_ohe, capacity, operating_agency_ohe], how='left')
    dummy_cols = [c for c in attribute_df.columns if c != 'CAP_MCM']
    attribute_df[dummy_cols] = attribute_df[dummy_cols].fillna(0)
    attribute_df.index = attribute_df.index.astype(index_type)
    return attribute_df

def data_processing(res_id, transform_type, left, right='2020-12-31', train_frac=0.6, val_frac=0.2, test_frac=0.2, return_scaler=False, storage=False, attributes=None):
    """
    Run data processing pipeline for one ResOPS reservoir.
    Params:
    res_id -- int, ResOPS reservoir ID
    transform_type -- str, in preprocessing, whether to 'standardize' or 'normalize' the data
    left -- str (YYYY-MM-DD), beginning boundary of time window
    right -- str (YYYY-MM-DD), end boundary of time window
    return_scaler -- bool, whether or not to return src.data.data_processing.time_scaler() object
    storage -- bool, whether or not to include storage data in features
    attributes -- pd.DataFrame, dataframe of reservoir attributes to include as features
    """

    # Read in data, columns are [inflow, outflow, storage]
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
    # Add day of the year (doy) as another column
    df['doy'] = df.index.to_series().dt.dayofyear.astype('float')
    # Add reservoir attributes if provided
    if attributes is not None:
        attr = attributes.loc[[res_id]]
        attr = pd.concat([attr]*len(df), ignore_index=True)
        attr.index = df.index
        df = pd.concat([df, attr], axis=1)
    # Select data window
    df = df[left:right].copy()

    # Get input feature columns. Static continuous attributes are appended after time-series preprocessing
    # so they are not rescaled separately within each reservoir.
    base_input_cols = ['inflow', 'doy'] + (['storage'] if storage else [])
    attribute_cols = [c for c in df.columns if c not in base_input_cols + ['storage', 'outflow']]
    static_continuous_cols = [c for c in ['CAP_MCM'] if c in attribute_cols]
    processed_attr_cols = [c for c in attribute_cols if c not in static_continuous_cols]
    processed_input_cols = base_input_cols + processed_attr_cols
    df_processed = df[processed_input_cols + ['outflow']].copy()

    # Get output target column index
    target_idx = [len(processed_input_cols)]


    # Run data processing pipeline
    pipeline = processing_pipeline(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, chunk_size=3*365, pad_value=-1, transform_type=transform_type, fill_na_method='mean')
    # Train/val/test tensors of shape (#chunks, chunksize, [inflow, outflow, storage, doy])
    ts_train, ts_val, ts_test = pipeline.process_data(df_processed) 

    # Separate inputs(X) and targets (y)
    X_train, X_val, X_test = ts_train[:, :, :len(processed_input_cols)], ts_val[:, :, :len(processed_input_cols)], ts_test[:, :, :len(processed_input_cols)]
    y_train, y_val, y_test = ts_train[:, :, target_idx], ts_val[:, :, target_idx], ts_test[:, :, target_idx]

    if static_continuous_cols:
        static_attr = torch.tensor(df[static_continuous_cols].iloc[0].values, dtype=torch.float).view(1, 1, -1)
        X_train = torch.cat((X_train, static_attr.expand(X_train.shape[0], X_train.shape[1], -1)), dim=2)
        X_val = torch.cat((X_val, static_attr.expand(X_val.shape[0], X_val.shape[1], -1)), dim=2)
        X_test = torch.cat((X_test, static_attr.expand(X_test.shape[0], X_test.shape[1], -1)), dim=2)

    if return_scaler:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), pipeline.scaler
    else:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class multi_reservoir_data_oos:
    """Store and combine data from multiple in sample and out of sample reservoirs"""
    def __init__(self, left_years_dict, res_list, oos_list, storage=False, attributes=None):
        """ 
        Params:
        left_years_dict: dict, dictionary of year of first available data from each requested reservoir (name : year)
        res_list: list of ResOps reservoir ID's of interest
        oos_list: list of out-of-sample ResOps reservoir ID's (subset of res_list)
        storage: bool, whether or not to include storage data as a feature in data processing (default False)
        attributes: pd.DataFrame, dataframe of reservoir attributes to include as features
        """
        self.left_years_dict = left_years_dict
        self.res_list = res_list
        self.oos_list = oos_list # out of sample reservoirs
        self.is_list = [item for item in res_list if item not in oos_list] # in sample reservoirs
        self.attributes = attributes
        self.storage = storage

        # For in-sample reservoirs: collect train and val tensors and their respective src.data.data_processing.time_scaler() objects
        self.X_train_dict = {}
        self.y_train_dict = {}
        self.X_val_dict = {}
        self.y_val_dict = {}
        self.scaler_dict_is = {}

        # For out-of-sample reservoirs: collect test tensors (full history) and their respective src.data.data_processing.time_scaler() objects
        self.X_test_dict = {}
        self.y_test_dict = {}
        self.scaler_dict_oos = {}
        return
    
    def fetch_data(self):
        """Fetch data for each reservoir. For in-sample reservoirs: split into train/val tensors. For oos reservors: reshape data into test tensors"""
        # Run data processing for each reservoir
        for reservoir, left_year in tqdm(self.left_years_dict.items(), desc='Processing data: '):
            # Out-of-sample reservoirs
            if reservoir in self.oos_list:
                result = data_processing(res_id=reservoir, transform_type='standardize', train_frac=1, val_frac=0, test_frac=0,
                                    left=f'{left_year}-01-01', right='2020-12-31',
                                    return_scaler=True, storage=self.storage, attributes=self.attributes)
                self.X_test_dict[reservoir] = result[0][0]
                self.y_test_dict[reservoir] = result[0][1]
                self.scaler_dict_is[reservoir] = result[3]

            # In-sample reservoirs
            else:
                result = data_processing(res_id=reservoir, transform_type='standardize', train_frac=0.75, val_frac=0.25, test_frac=0,
                                        left=f'{left_year}-01-01', right='2020-12-31',
                                        return_scaler=True, storage=self.storage, attributes=self.attributes)
                # Save results
                self.X_train_dict[reservoir] = result[0][0] # (# chunks, chunk size, # features (e.g. inflow and doy))
                self.y_train_dict[reservoir] = result[0][1] # (# chunks, chunk size, 1 (outflow))
                self.X_val_dict[reservoir] = result[1][0]
                self.y_val_dict[reservoir] = result[1][1]
                self.scaler_dict_oos[reservoir] = result[3]
        return
    
    def combine_reservoir_data(self):
        """ 
        Concatenate all fetched reservoir data into one train, val, test tensor
        """
        # Concat tensors along chunks dimension (dim = 0)
        X_train, y_train = torch.cat([self.X_train_dict[key] for key in self.is_list], dim=0), torch.cat([self.y_train_dict[key] for key in self.is_list], dim=0)
        X_val, y_val = torch.cat([self.X_val_dict[key] for key in self.is_list], dim=0), torch.cat([self.y_val_dict[key] for key in self.is_list], dim=0)
        X_test, y_test = torch.cat([self.X_test_dict[key] for key in self.oos_list], dim=0), torch.cat([self.y_test_dict[key] for key in self.oos_list], dim=0)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def fetch_and_combine(self):
        """Run fetch_data and return combined tensors"""
        self.fetch_data()
        return self.combine_reservoir_data()
    
def train_simultaneous_model(X_train, y_train, X_val, y_val, num_layers=1,hidden_size1=30, hidden_size2=15,
                             dropout_prob=0.3, random_seed=0, plot=False, device=torch.device("cpu")):
    """
    Train simultanoues LSTM model on in-sample reservoirs
    Params:
    X_train/X_val -- train/val input tensors of shape (# batches, batch size, # features)
    y_train/y_val -- train/val target tensors of shape (# batches, batch size, 1)
    plot -- bool, whether or not to plot train/val losses
    """
    # Create PyTorch Dataset and Dataloader from training and validation for in-sample reservoirs
    dataset_train, dataset_val = (TensorDataset(X_train, y_train), TensorDataset(X_val, y_val))
    dataloader_train, dataloader_val = (DataLoader(dataset_train, batch_size=256, shuffle=False), 
                                        DataLoader(dataset_val, batch_size=256, shuffle=False))
    # Instantiate model (Model 1a archeticture)
    input_size = X_train.shape[2]
    output_size = 1
    torch.manual_seed(random_seed)
    model = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, num_layers=num_layers, dropout_prob=dropout_prob)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Run training loop
    train_losses, val_losses = training_loop(model=model, criterion=criterion, optimizer=optimizer, 
                                            patience=10, dataloader_train=dataloader_train, 
                                            dataloader_val=dataloader_val, epochs=1000, device=device)
    if plot:
        plt.figure()
        plot_train_val(train_losses=train_losses, val_losses=val_losses)
        plt.show()

    return train_losses, val_losses, model

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
        'reviewer_comments',
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
    grid_search_df.to_csv(f'report/results/reviewer_comments/hyperparameter_tuning_pooled/grid_search_model1_pooled.csv')


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
    torch.save(simul_model.state_dict(), 'report/results/reviewer_comments/hyperparameter_tuning_pooled/resops_simul_model.pt')
    
    # -----     Evaluate and save in-sample performance     ----- #

    simul_model.to(torch.device("cpu")) # move model to CPU for evaluation
    r2_in_sample_df = pd.DataFrame(index=data_combiner.is_list, columns=['train', 'val'])
    for in_res in tqdm(data_combiner.is_list, desc='Evaluating in-sample performance: '):
        r2_in_sample_df.loc[in_res, :] = [r2_score_tensor(model=simul_model, X=data_combiner.X_train_dict[in_res], 
                                                          y=data_combiner.y_train_dict[in_res]),
                                          r2_score_tensor(model=simul_model, X=data_combiner.X_val_dict[in_res], 
                                                          y=data_combiner.y_val_dict[in_res])]
    r2_in_sample_df.to_csv('report/results/reviewer_comments/hyperparameter_tuning_pooled/resops_oos_in_sample_train_val.csv')

    # -----     Evaluate and save out-of-sample performance     ----- #

    r2_out_sample_df = pd.DataFrame(index=data_combiner.oos_list, columns=['test'])
    for out_res in tqdm(data_combiner.oos_list, desc='Evaluating out-of-sample performance: '):
        r2_out_sample_df.loc[out_res, :] = r2_score_tensor(model=simul_model, X=data_combiner.X_test_dict[out_res], y=data_combiner.y_test_dict[out_res])
    r2_out_sample_df.to_csv('report/results/reviewer_comments/hyperparameter_tuning_pooled/resops_oos_out_of_sample_test.csv')

    return

# run script
if __name__ == '__main__':
    main()
    


