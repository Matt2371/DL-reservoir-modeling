### FUNCTIONS TO SUPPORT DATA PROCESSING USING AUTOREGRESSIVE LAGS ###
### USEFUL FOR NON-LSTM BENCHMARK MODELS ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.data_processing import *
from src.data.data_fetching import *
from sklearn.preprocessing import StandardScaler

def create_lags(df, n_lags, exclude_list):
    """ 
    Add autoregressive lags as columns to a dataframe
    Params:
    df: pandas df, input data
    n_lags: int
    exclude_list: list, str of column names to not create lags for
    """
    df = df.copy()
    for column in df.columns:
        if column in exclude_list:
            continue
        for lag in range(n_lags):
            df[f'{column}_lag{lag+1}'] = df[column].shift(lag)
    return df

class processing_pipeline_w_lags:
    """
    Run data processing pipeline:
    1. Add autoregressive lags
    2. Conduct train/val/test split
    3. Fill NA with training mean
    4. Standardize data with training stats
    """
    def __init__(self, df, n_lags, exclude_list, train_frac=0.6, val_frac=0.2, test_frac=0.2, left_year=1944, right_year=2022):
        """ 
        Params:
        df: pandas df, input data
        *params for create_lags()
        *params for train/val/test split
        left_year: int, year for left data window (1944 for Shasta)
        right_year: int, year for right data window (2022 by default)
        """
        self.df = df # Store original data
        self.n_lags = n_lags
        self.exclude_list = exclude_list
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.left_year = left_year
        self.right_year = right_year
        self.df_lagged = None # Store df with added lags
        self.scaler = StandardScaler()
        
        # Store processed data
        self.df_train = None
        self.df_val = None
        self.df_test = None
        return
    
    def process_data(self):
        # Add lags
        self.df_lagged = create_lags(self.df, n_lags=self.n_lags, exclude_list=self.exclude_list)
        # Trim leading NA due to creating lags
        self.df_lagged = self.df_lagged[f'{self.left_year}-01-0{self.n_lags}':f'{self.right_year}-12-31']

        # Train / val / test split
        df_train, df_val, df_test = train_val_test(data=self.df_lagged, train_frac=self.train_frac, val_frac=self.val_frac, test_frac=self.test_frac)

        # Fill NA with training mean
        df_train = df_train.fillna(df_train.mean())
        df_val = df_val.fillna(df_train.mean())
        df_test = df_test.fillna(df_train.mean())

        # Standardize data
        df_train = self.scaler.fit_transform(df_train)
        df_val = self.scaler.transform(df_val)
        df_test = self.scaler.transform(df_test)

        # Save and return results
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        return df_train, df_val, df_test