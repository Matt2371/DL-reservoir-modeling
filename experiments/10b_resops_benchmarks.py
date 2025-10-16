### TRAIN AND EVALUATE BENCHMARK MODELS ON RESOPS DATASET RESERVOIRS INDIVIDUALLY ###
### MODEL RELEASE AGAINST 5 INFLOW LAGS AND CURRENT STORAGE ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from src.data.data_processing import *
from src.data.data_fetching import *
from src.data.data_processing_lagged import *
from src.models.predict_model_lagged import *

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def data_processing(res_id, left_year, right_year=2020, train_frac=0.6, val_frac=0.2, test_frac=0.2, log_names=[], return_scaler=False, storage=False):
    """
    Run data processing pipeline for one ResOPS reservoir.
    Params:
    res_id -- int, ResOPS reservoir ID
    transform_type -- str, in preprocessing, whether to 'standardize' or 'normalize' the data
    left_year -- int, beginning year boundary of time window
    right_year -- int, end year boundary of time window
    log_names -- list of column names (str) to take log of before running rest of pipeline. E.g. ['inflow', 'outflow', 'storage']
    return_scaler -- bool, whether or not to return src.data.data_processing.time_scaler() object
    storage -- bool, whether or not to include storage data in features
    """

    # Read in data, columns are [inflow, outflow, storage]
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
    # Add day of the year (doy) as another column
    df['doy'] = df.index.to_series().dt.dayofyear
    # Select data window
    df = df[f'{left_year}-01-01':f'{right_year}-12-31'].copy()

    # Take log of df columns that are in log_names
    for column_name in df.columns:
        if column_name in log_names:
            df[column_name] = np.log(df[column_name])
        else:
            continue

    # Run data processing pipeline
    pipeline = processing_pipeline_w_lags(df=df, n_lags=5, exclude_list=['outflow', 'storage', 'doy'], 
                                          train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, left_year=left_year, right_year=right_year)
    # Train/val/test arrays of shape (timesteps, ['inflow', 'outflow', 'storage', 'doy', 'inflow_lag1', 'inflow_lag2', 'inflow_lag3', 'inflow_lag4', 'inflow_lag5'])
    df_train, df_val, df_test = pipeline.process_data() 

    # Separate inputs(X) and targets (y)
    if storage:
        X_train, X_val, X_test = df_train[:, np.r_[0, 2:9]], df_val[:, np.r_[0, 2:9]], df_test[:, np.r_[0, 2:9]]
    else:
        X_train, X_val, X_test = df_train[:, np.r_[0, 3:9]], df_val[:, np.r_[0, 3:9]], df_test[:, np.r_[0, 3:9]]
    # select outflow as target feature
    y_train, y_val, y_test = df_train[:, 1], df_val[:, 1], df_test[:, 1]

    if return_scaler:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), pipeline.scaler
    else:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
def train_one_reservoir(res_id, left_year, type):
    """ 
    Train benchmark lagged model for one ResOPS reservoir, return R2 scores for train/val/test
    Params:
    res_id: int, ResOPS reservoir ID
    left_year: int, year corresponding to data left window
    type: str, type of model {'linear', 'random_forest'}
    """
    # Run data processing pipeline (resulting tuple contains (X, y)), include storage for lagged model
    train_tuple, val_tuple, test_tuple = data_processing(res_id=res_id, left_year=left_year, return_scaler=False, storage=True)

    # Instantiate model
    if type == 'linear':
        model = LinearRegression()
    elif type == 'random_forest':
        model = RandomForestRegressor(max_depth=5)

    # Train model
    model.fit(*train_tuple)
    
    # Evaluate train/val/test R2 score
    r2_train, r2_val, r2_test = eval_train_val_test_lagged(model=model, X_train=train_tuple[0], X_val=val_tuple[0], X_test=test_tuple[0],
                                                           y_train=train_tuple[1], y_val=val_tuple[1], y_test=test_tuple[1])
    return r2_train, r2_val, r2_test

def train_all_reservoirs(res_list, left_years_dict, type):
    """ 
    Train benchmark lagged model for all selected ResOPS reservoirs.
    Return df of resulting train/val/test R2
    Params: 
    res_list -- list, list of ResOPS ID's to train models for
    left_years_dict -- dict, maps reservoir ID in res_list to left data window year
    type: str, type of model {'linear', 'random_forest'}
    """
    # Initialize df to store results
    df = pd.DataFrame(index=res_list, columns=['train', 'val', 'test'])

    for res in tqdm(res_list, desc=f'Training {type} models: '):
        # Train models for each reservoir
        r2_train, r2_val, r2_test = train_one_reservoir(res_id=res, left_year=left_years_dict[res], type=type)
        # Save results
        df.loc[res, :] = (r2_train, r2_val, r2_test)

    return df

def main():
    # Filter reservoirs by record length (90% data record complete)
    res_list = filter_res()

    # Get data window left year for each filtered reservoir
    left_years_dict = get_left_years(res_list=res_list)

    # Train linear model for each reservoir, save r2 metrics
    df_linear_r2 = train_all_reservoirs(res_list=res_list, left_years_dict=left_years_dict, type='linear')

    # Train random forest model for each reservoir, save r2 metrics
    df_rf_r2 = train_all_reservoirs(res_list=res_list, left_years_dict=left_years_dict, type='random_forest')

    # Save results
    df_linear_r2.to_csv('report/results/resops_training/resops_benchmark_linear_r2.csv')
    df_rf_r2.to_csv('report/results/resops_training/resops_benchmark_rf_r2.csv')
    return

# run script
if __name__ == '__main__':
    main()
    