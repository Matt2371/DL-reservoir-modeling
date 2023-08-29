import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

### FUNCTIONS AND CLASSES TO ASSIST WITH DATA PROCESSING ###

def train_val_test(data, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    """
    Split the data into training, validation, and test sets.
    Parameters:
    data -- 2darray (time, features).
    train_frac, val_frac, test_frac -- proportion of the data for each set
    Returns: 
    (train, val, test) datasets of shape (timesteps, features)
    """
    assert train_frac + val_frac + test_frac == 1

    timesteps = data.shape[0] # length of input timeseries
    train_size = int(round(timesteps * train_frac))
    val_size = int(round(timesteps * val_frac))

    train, val, test = data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]
    return train, val, test

class time_standardizer:
    """Standardize timeseries data (2darray) using statistics from the training data"""
    def __init__(self):
        # statistics are calculated for each feature across all observations and timesteps
        self.mean = 0
        self.std = 0
        self.isfit = False
    
    def fit(self, train_data):
        """
        Calculates statistics from the train data to use in standardization
        Params:
        train_data -- 2darray of shape (timesteps, features)
        """
        # use pandas so nan is ignored when calculating summary statistics
        train_data = pd.DataFrame(train_data)
        # calculate normalizing statistics
        self.mean = train_data.mean(axis=0).values.reshape(1, -1) # (1, features)
        self.std = train_data.std(axis=0).values.reshape(1, -1) # (1, features)
        self.isfit = True
        return
    
    def transform(self, data):
        """Transforms the input data after fitting"""
        assert self.isfit == True
        result = (data - self.mean) / self.std # (timesteps, features)
        return result
    
    def inverse_transform(self, data):
        """Inverses a transformation"""
        assert self.isfit == True
        result = data * self.std + self.mean # (timesteps, features)
        return result
    

def fill_nan(data):
    """Fill nan with 0: equilvalent to filling na using mean from the training data if data is standardized"""
    data = pd.DataFrame(data)
    data.fillna(value=0, inplace=True)
    return data.values

def split_and_pad(data, chunk_size=3*365, pad_value=-1, pad_nan=False):
    """
    Splits data into chunks (along time dimension) and pads them so each chunk is of the same length.
    Also pads nan values.
    Splitting a timeseries into chunks may provide more stable training. 
    Parameters:
    data -- input data, tensor of shape (timesteps, num_features)
    chunk_size -- int, number of timesteps in each chunk
    pad_value -- int, number that represents padding or nan values
    pad_nan -- bool, whether or not to also pad nan with pad value
    Returns: 
    Tensor of shape (num_splits, chunk_size, num_features)
    """

    # Split the data into chunks (each chunks is of shape (chunk_size, num_features))
    data_splits = torch.split(data, chunk_size, dim=0)

    # Pad each chunk so that they are the same size
    data_new = pad_sequence(data_splits, batch_first=True, padding_value=pad_value).to(torch.float) 

    # Pad nan
    if pad_nan:
        data_new[data_new.isnan()] = pad_value

    return data_new # output is of shape (num_splits, chunk_size, num_features)

class processing_pipeline():
    """
    Pipeline for processing data:
    1. Split data into train/val/test sets
    2. Standardize the data using the training set
    3. Fill nan with 0 (equivalent to filling with mean from train data)
    4. Convert data to Pytorch tensors
    5. Split data into chunks and pad the remainder
    """
    def __init__(self, train_frac=0.6, val_frac=0.2, test_frac=0.2, chunk_size=3*365, pad_value=-1):
        """ 
        Params:
        train_frac, val_frac, test_frac -- fractions to partition train/val/test split
        chunk_size -- size of each chunk when splitting data
        pad_value -- value to pad the remainder after splitting data into chunks
        """
        # save pipeline parameters
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.chunk_size = chunk_size
        self.pad_value = pad_value

        # instantiate scaler object
        self.scaler = time_standardizer() 

    def process_data(self, data):
        """
        Process raw input data of shape (timesteps, features)
        Returns: (train, val, test) processed PyTorch tensors of shape (#chunks, chunk_size, #features)
        """
        # 1. Split into training/validation/testing sets
        df_train, df_val, df_test = train_val_test(data=data, train_frac=self.train_frac, val_frac=self.val_frac, test_frac=self.test_frac)
        # 2. Standardize the data using the training set
        self.scaler.fit(df_train)
        df_train, df_val, df_test = self.scaler.transform(df_train), self.scaler.transform(df_val), self.scaler.transform(df_test)
        # 3. Fill nan with 0 (equivalent to filling with mean from train data)
        df_train, df_val, df_test = fill_nan(df_train), fill_nan(df_val), fill_nan(df_test)
        # 4. Convert data to PyTorch tensors (from numpy array)
        ts_train, ts_val, ts_test = torch.tensor(df_train, dtype=torch.float), torch.tensor(df_val, dtype=torch.float), torch.tensor(df_test, dtype=torch.float)
        # 5. Split data into chunks and pad the remainder (for training and validation sets)
        ts_train, ts_val = split_and_pad(ts_train), split_and_pad(ts_val)

        return ts_train, ts_val, ts_test