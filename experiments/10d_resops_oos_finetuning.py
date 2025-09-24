### STUDY FINETUNING THE POOLED MODEL FROM EXPERIMENT 10d ###
### TRAIN (75) / VAL (25) SPLIT ON FINETUNING DATA (FIRST N YEARS) ###
### COMPARE RESULTS OF INDIVIDUAL MODEL, POOLED MODEL, 
### AND FINETUNED MODEL ON LAST 20% OF RECORD (SAME TEST SET AS OTHER EXPERIMENTS) ###

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
import geopandas as gpd
from math import floor
import os
import copy
from tqdm import tqdm

from src.data.data_processing import *
from src.data.data_fetching import *
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import *

def get_attributes(index_type=str):
    """
    Get one-hot encorded reservoir attributes df (main use from GRanD, DOR category from mean-inflow/max-storage or GRanD DOR_PC)
    Params:
    index_type -- type of index for returned dataframe, either str or int
    """
    # GRanD Attributes
    gdf = gpd.read_file("data/GRanD/GRanD_dams_v1_3.shp")
    gdf = gdf.drop(columns="geometry").set_index("GRAND_ID")

    # Main reservoir use
    use_ohe = pd.get_dummies(gdf['MAIN_USE'], prefix='USE', dtype='float')
    use_ohe.index = use_ohe.index.astype(str)

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

    attribute_df = use_ohe.join(dor_ohe, how='left')
    attribute_df.index = attribute_df.index.astype(index_type)
    return attribute_df


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
    attributes -- pd.DataFrame, dataframe of one-hot encoded reservoir attributes to include as features
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

    # Get input feature column indices (including one-hot attribute features if provided)
    base_input_cols = ['inflow', 'doy'] + (['storage'] if storage else [])
    is_binary = (df.isin([0, 1]) | df.isna()).all(axis=0)     
    one_hot_cols = [c for c, b in is_binary.items() if b and c not in base_input_cols and c != 'outflow']
    input_idx = [df.columns.get_loc(c) for c in (base_input_cols + one_hot_cols)]

    # Get output target column index
    target_idx = [df.columns.get_loc('outflow')]


    # Run data processing pipeline
    pipeline = processing_pipeline(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, chunk_size=3*365, pad_value=-1, transform_type=transform_type, fill_na_method='mean')
    # Train/val/test tensors of shape (#chunks, chunksize, [inflow, outflow, storage, doy])
    ts_train, ts_val, ts_test = pipeline.process_data(df) 

    # Separate inputs(X) and targets (y)
    X_train, X_val, X_test = ts_train[:, :, input_idx], ts_val[:, :, input_idx], ts_test[:, :, input_idx]
    y_train, y_val, y_test = ts_train[:, :, target_idx], ts_val[:, :, target_idx], ts_test[:, :, target_idx]

    if return_scaler:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), pipeline.scaler
    else:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    

class multi_reservoir_data:
    """Store and combine data from multiple in sample and out of sample reservoirs"""
    def __init__(self, left_years_dict, res_list, attributes=None):
        """ 
        Params:
        left_years_dict: dict, dictionary of year of first available data from each requested reservoir (name : year)
        res_list: list of ResOps reservoir ID's of interest
        attributes: pd.DataFrame, dataframe of one-hot encoded reservoir attributes to include as features
        """
        self.left_years_dict = left_years_dict
        self.res_list = res_list
        self.attributes = attributes

        self.X_train_dict = {}
        self.y_train_dict = {}
        self.X_val_dict = {}
        self.y_val_dict = {}
        self.X_test_dict = {}
        self.y_test_dict = {}
        self.scaler_dict = {}
        return
    
    def fetch_data(self):
        # Run data processing for each reservoir
        for reservoir, left_year in tqdm(self.left_years_dict.items(), desc='Processing data: '):
            result = data_processing(res_id=reservoir, transform_type='standardize', train_frac=0.6, val_frac=0.2, test_frac=0.2,
                                    left=f'{left_year}-01-01', right='2020-12-31',
                                    return_scaler=True, attributes=self.attributes)
            # Save results
            self.X_train_dict[reservoir] = result[0][0] # (# chunks, chunk size, # features (e.g. inflow and doy))
            self.y_train_dict[reservoir] = result[0][1] # (# chunks, chunk size, 1 (outflow))
            self.X_val_dict[reservoir] = result[1][0]
            self.y_val_dict[reservoir] = result[1][1]
            self.X_test_dict[reservoir] = result[2][0]
            self.y_test_dict[reservoir] = result[2][1]
            self.scaler_dict[reservoir] = result[3]
        return

def finetune_first_nyears(res_id, left_year, nyears, attributes=None, baseline=False):
    """
    Finetune trained pooled model (from experiment 10c) 
    based on the first n years of data
    Params:
    res_id: reservoir to finetune to
    left_year: left year of data window, i.e. first year of data record
    nyears: first n years of data from reservoir to use for finetuning
    attributes: pd.DataFrame, dataframe of one-hot encoded reservoir attributes to include as features (in data processing)
    baseline: bool, if True, do not finetune, train new model from scratch
    Returns:
    finetuned_model
    """
    # Load multi-reservoir model, instantiate loss and optimizer
    if attributes is not None:
        input_size = 2 + attributes.shape[1]  # Add number of one-hot encoded attributes to input size
    else:
        input_size = 2  # inflow and doy only
    hidden_size1 = 30
    hidden_size2 = 15
    output_size = 1
    dropout_prob = 0.3
    num_layers = 1
    torch.manual_seed(0)
    model = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, 
                                num_layers=num_layers, dropout_prob=dropout_prob)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if not baseline:
        model.load_state_dict(torch.load('src/models/saved_models/resops_simul_model.pt', weights_only=True))

    # Get train and validation data for first n years of data
    data_result = data_processing(res_id=res_id, transform_type='standardize', 
                                  train_frac=0.75, val_frac=0.25, test_frac=0,
                                  left=f'{left_year}-01-01', right=f'{left_year + nyears - 1}-12-31',
                                  return_scaler=True, attributes=attributes)
    dataset_train, dataset_val = (TensorDataset(*data_result[0]), TensorDataset(*data_result[1]))
    dataloader_train, dataloader_val = (DataLoader(dataset_train, batch_size=1, shuffle=False), 
                                        DataLoader(dataset_val, batch_size=1, shuffle=False))
    
    # (Finetuning) training loop
    train_losses, val_losses = training_loop(model=model, criterion=criterion, optimizer=optimizer, 
                                            patience=10, dataloader_train=dataloader_train, 
                                            dataloader_val=dataloader_val, epochs=1000)

    return model

def main():
    # Get one-hot encoded reservoir attributes dataframe
    attribute_df = get_attributes(index_type=int)

    # Read list of out-of-sample reservoirs from experiment 10c, get left years dictionary
    res_list = pd.read_csv('report/results/resops_training/resops_oos_out_of_sample_test.csv', index_col=0).index.to_list()
    left_year_dict = get_left_years(res_list=res_list)
    
    # Get final 20% of data record as test set, initialize dataframe to store results comparing
    # individual training, multi-reservoir model, and finetuning
    # Recall that 60/20/20 was the default train/val/test set, so we can just extract the test data
    complete_data_record = multi_reservoir_data(left_years_dict=left_year_dict, res_list=res_list, attributes=attribute_df)
    complete_data_record.fetch_data()
    X_test_dict = complete_data_record.X_test_dict
    y_test_dict = complete_data_record.y_test_dict
    final_results = pd.DataFrame(index=res_list, columns=['individual',
                                                          'pooled',
                                                          'finetuned_pooled_5yr',
                                                          'finetuned_pooled_10yr',
                                                          'finetuned_pooled_15yr',
                                                          'finetuned_pooled_20yr',
                                                          'finetuned_pooled_25yr',
                                                          'finetuned_pooled_30yr'])
    baseline_results = pd.DataFrame(index=res_list, columns=['baseline_5yr',
                                                              'baseline_10yr',
                                                              'baseline_15yr',
                                                              'baseline_20yr',
                                                              'baseline_25yr',
                                                              'baseline_30yr'])
    
    # Get individual model R2 on last 20% of record (test set)
    individual_r2 = pd.read_csv('report/results/resops_training/resops_individual_r2.csv', index_col=0)
    final_results.loc[res_list, 'individual'] = individual_r2.loc[res_list, 'test']
    
    # Get pooled model R2 on last 20% of record (test set)
    # input_size = 2
    input_size = 2 + attribute_df.shape[1]  # Add number of one-hot encoded attributes to input size
    hidden_size1 = 30
    hidden_size2 = 15
    output_size = 1
    dropout_prob = 0.3
    num_layers = 1
    torch.manual_seed(0)
    model_pooled = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, 
                                num_layers=num_layers, dropout_prob=dropout_prob)
    model_pooled.load_state_dict(torch.load('src/models/saved_models/resops_simul_model.pt', weights_only=True))
    for res in res_list:
        final_results.loc[res, 'pooled'] = r2_score_tensor(model=model_pooled,
                                                                X=X_test_dict[res],
                                                                y=y_test_dict[res])
        
    # Get finetuned model R2 on last 20% of record (test set)
    finetune_year_list = [5, 10, 15, 20, 25, 30]
    for first_nyears in finetune_year_list:
        for res in res_list:
            # Finetune model to res
            finetuned_model = finetune_first_nyears(res_id=res, left_year=left_year_dict[res], nyears=first_nyears, attributes=attribute_df)
            finetuned_model.to(torch.device("cpu")) # Move model to CPU for evaluation
            # Finetuned R2 on test
            final_results.loc[res, f'finetuned_pooled_{first_nyears}yr'] = r2_score_tensor(model=finetuned_model,
                                                                                           X=X_test_dict[res],
                                                                                           y=y_test_dict[res])
            # Baseline model (no pretraining) R2 on test
            baseline_model = finetune_first_nyears(res_id=res, left_year=left_year_dict[res], nyears=first_nyears, attributes=attribute_df, baseline=True)
            baseline_model.to(torch.device("cpu")) # Move model to CPU for evaluation
            baseline_results.loc[res, f'baseline_{first_nyears}yr'] = r2_score_tensor(model=baseline_model,
                                                                                      X=X_test_dict[res],
                                                                                      y=y_test_dict[res])
    # Save final results
    final_results.to_csv('report/results/resops_training/resops_oos_finetuning.csv')
    baseline_results.to_csv('report/results/resops_training/resops_oos_finetuning_baseline.csv')
    return

# run script
if __name__ == '__main__':
    main()
