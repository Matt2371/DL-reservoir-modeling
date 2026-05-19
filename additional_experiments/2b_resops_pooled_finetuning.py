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

from additional_experiments.pooled_training_utils import get_attributes, data_processing, multi_reservoir_data

def finetune_first_nyears(res_id, left_year, nyears, attributes=None, baseline=False, device=torch.device("cpu")):
    """
    Finetune trained pooled model (from experiment 10c) 
    based on the first n years of data
    Params:
    res_id: reservoir to finetune to
    left_year: left year of data window, i.e. first year of data record
    nyears: first n years of data from reservoir to use for finetuning
    attributes: pd.DataFrame, dataframe of reservoir attributes to include as features (in data processing)
    baseline: bool, if True, do not finetune, train new model from scratch
    device: torch.device to train model
    Returns:
    finetuned_model
    """
    # Get train and validation data for first n years of data
    data_result = data_processing(res_id=res_id, transform_type='standardize', 
                                  train_frac=0.75, val_frac=0.25, test_frac=0,
                                  left=f'{left_year}-01-01', right=f'{left_year + nyears - 1}-12-31',
                                  return_scaler=True, attributes=attributes)

    # Load multi-reservoir model, instantiate loss and optimizer
    input_size = data_result[0][0].shape[2]
    hidden_size1 = 80
    hidden_size2 = 120
    output_size = 1
    dropout_prob = 0.3
    num_layers = 2
    torch.manual_seed(0)
    model = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, 
                                num_layers=num_layers, dropout_prob=dropout_prob)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if not baseline:
        model.load_state_dict(torch.load('report/results/additional_experiments/hyperparameter_tuning_pooled/resops_simul_model.pt', weights_only=True))

    dataset_train, dataset_val = (TensorDataset(*data_result[0]), TensorDataset(*data_result[1]))
    dataloader_train, dataloader_val = (DataLoader(dataset_train, batch_size=1, shuffle=False), 
                                        DataLoader(dataset_val, batch_size=1, shuffle=False))
    
    # (Finetuning) training loop
    train_losses, val_losses = training_loop(model=model, criterion=criterion, optimizer=optimizer, 
                                             patience=10, dataloader_train=dataloader_train, 
                                             dataloader_val=dataloader_val, epochs=1000, device=device)

    return model

def main():
    # Set the device
    device = get_device()
    print(f"Using device: {device}")

    # Get reservoir attributes dataframe
    attribute_df = get_attributes(index_type=int)

    # Read list of out-of-sample reservoirs from additional_experiments 2a, get left years dictionary
    res_list = pd.read_csv('report/results/additional_experiments/hyperparameter_tuning_pooled/resops_oos_out_of_sample_test.csv', index_col=0).index.to_list()
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
    input_size = next(iter(X_test_dict.values())).shape[2]
    hidden_size1 = 80
    hidden_size2 = 120
    output_size = 1
    dropout_prob = 0.3
    num_layers = 2
    torch.manual_seed(0)
    model_pooled = LSTMModel1_opt(input_size=input_size, hidden_size1=hidden_size1, 
                                hidden_size2=hidden_size2, output_size=output_size, 
                                num_layers=num_layers, dropout_prob=dropout_prob)
    model_pooled.load_state_dict(torch.load('report/results/additional_experiments/hyperparameter_tuning_pooled/resops_simul_model.pt', weights_only=True))
    for res in res_list:
        final_results.loc[res, 'pooled'] = r2_score_tensor(model=model_pooled,
                                                                X=X_test_dict[res],
                                                                y=y_test_dict[res])
        
    # Get finetuned model R2 on last 20% of record (test set)
    finetune_year_list = [5, 10, 15, 20, 25, 30]
    for first_nyears in finetune_year_list:
        for res in res_list:
            # Finetune model to res
            finetuned_model = finetune_first_nyears(res_id=res, left_year=left_year_dict[res], nyears=first_nyears, attributes=attribute_df, device=device)
            finetuned_model.to(torch.device("cpu")) # Move model to CPU for evaluation
            # Finetuned R2 on test
            final_results.loc[res, f'finetuned_pooled_{first_nyears}yr'] = r2_score_tensor(model=finetuned_model,
                                                                                           X=X_test_dict[res],
                                                                                           y=y_test_dict[res])
            # Baseline model (no pretraining) R2 on test
            baseline_model = finetune_first_nyears(res_id=res, left_year=left_year_dict[res], nyears=first_nyears, attributes=attribute_df, baseline=True, device=device)
            baseline_model.to(torch.device("cpu")) # Move model to CPU for evaluation
            baseline_results.loc[res, f'baseline_{first_nyears}yr'] = r2_score_tensor(model=baseline_model,
                                                                                      X=X_test_dict[res],
                                                                                      y=y_test_dict[res])
    # Save final results
    final_results.to_csv('report/results/additional_experiments/hyperparameter_tuning_pooled/resops_oos_finetuning.csv')
    baseline_results.to_csv('report/results/additional_experiments/hyperparameter_tuning_pooled/resops_oos_finetuning_baseline.csv')
    return

# run script
if __name__ == '__main__':
    main()
