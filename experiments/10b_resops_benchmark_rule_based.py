### TRAIN AND EVALUATE BENCHMARK RULE BASED MODEL ON RESOPS DATASET RESERVOIRS INDIVIDUALLY ###

############# TRAIN AND EVALUATE SSJRB RESERVOIR MODEL ON RESOPS RESERVOIRS #####################
# Saves model parameters for each reservoir as well as csv of r2 scores in train/val/test for all reservoirs

import os
import sys

# Workaround: add directory of 'src' and 'ssjrb_wrapper' to the sys.path
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..')) # One level up to the project root
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ssjrb_wrapper.model_wrapper import reservoir_model
from ssjrb_wrapper.util import read_historical_df
from ssjrb_wrapper.util import water_day
from ssjrb_wrapper.train_preprocess import train_medians

from src.data.data_fetching import *
from src.data.data_processing import *
from src.models.model_zoo import LSTMModel1_opt
from src.models.train_model import *
from src.models.predict_model import *


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

def fit_one_reservoir(res_id, left, right = '2020-12-31'):
    """
    Fit SSJRB reservoir model to a single reservoir for ResOPS data

    Params:
    res_id -- str, Reservoir ID to fit model to
    left -- str (datelike), left window date to fetch data from
    right -- str (datelike), right window date to fetch data from

    Returns:
    model -- fitted SSJRB model
    r2_scores -- df of outflow r2 scores for train/val/test splits
    """

    ### DATA PROCESSING

    # Read in data, columns are [inflow, outflow, storage]
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
    # Select left-right window
    df = df[left:right].copy()

    # Convert units to work with SSJRB wrapper
    df[f'{res_id}_inflow_cfs'] = df['inflow'] * 35.3147 # m3/s to cfs
    df[f'{res_id}_outflow_cfs'] = df['outflow'] * 35.3147 # m3/s to cfs
    df[f'{res_id}_storage_af'] = df['storage'] * 810.713 # MCM to AF

    # Add DOWY column
    doy_series = df.index.to_series().dt.dayofyear
    df['dowy'] = [water_day(i) for i in doy_series]

    # Train/val/test split
    df_train, df_val, df_test = train_val_test(df, train_frac=0.6, val_frac=0.2, test_frac=0.2)

    # Fill missing values with mean of training data
    train_mean = df_train.mean()
    df_train = df_train.fillna(train_mean)
    df_val = df_val.fillna(train_mean)
    df_test = df_test.fillna(train_mean)


    ### MODEL TRAINING

    # Capacity (take as max of training storage series)
    training_capacity = df_train[f'{res_id}_storage_af'].max() / 1000 # units = TAF
    capacity_dict = {res_id: training_capacity}

    # Instantiate model
    model_i = reservoir_model(reservoir_capacity = capacity_dict)
    # Fit with early stopping on validation data
    model_i.fit(df_train=df_train, df_val=df_val, patience=5)

    ### EVALUATE MODEL AND SAVE PARAMS

    # Get predictions from fitted model, evaluate r2 score
    df_hat_train, df_hat_val, df_hat_test = (model_i.predict(df = df_train),
                                            model_i.predict(df = df_val),
                                            model_i.predict(df = df_test))

    r2_scores = pd.DataFrame({'train': r2_score(df_train[f'{res_id}_outflow_cfs'], df_hat_train[f'{res_id}_outflow_cfs']),
                            'val': r2_score(df_val[f'{res_id}_outflow_cfs'], df_hat_val[f'{res_id}_outflow_cfs']),
                            'test': r2_score(df_test[f'{res_id}_outflow_cfs'], df_hat_test[f'{res_id}_outflow_cfs'])}, index=[res_id])

    # Save model parameters
    model_i.save_params(filepath='./results/saved_models/resops_ssjrb_reservoir', fileprefix=f'resops_ssjrb_model_{res_id}')

    return model_i, r2_scores

def fit_all_reservoirs(res_list, left_years_dict):
    """
    Run fit_one_reservoir for all reservoirs
    Params:
    res_list -- list of ResOPS ID's to fit model to
    left_years_dict -- dict of left window years for each reservoir ID in res_list
    Returns:
    r2_scores_all -- df of outflow r2 scores for train/val/test splits for all reservoirs
    """
    # Initialize r2 scores df
    r2_scores_all = pd.DataFrame()

    # Fit and evaluate each reservoir in res_list
    for res_id in tqdm(res_list, desc='Fitting reservoirs: '):
        _, r2_score_i = fit_one_reservoir(res_id = res_id, left = f'{left_years_dict[res_id]}-01-01', right = '2020-12-31')
        r2_scores_all = pd.concat([r2_scores_all, r2_score_i], axis=0)

    return r2_scores_all

def main():
    # Get list of resops reservoirs, filtering by record length (90% data record complete)
    res_list = filter_res()

    # Get corresponding left years
    left_years_dict = get_left_years(res_list)
    
    # Train and evaluate models
    r2_scores_all = fit_all_reservoirs(res_list, left_years_dict)

    # Save r2 scores
    r2_scores_all.to_csv('report/results/resops_training/resops_benchmark_rule_based_r2.csv')
    return

# Run script
if __name__ == '__main__':
    main()