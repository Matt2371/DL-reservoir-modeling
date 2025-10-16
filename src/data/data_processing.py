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

class time_scaler():
    """
    Prepare timeseries data (2darray) for modeling by either:
    1. Standardize using (mean and std) from the training data
    2. Normalize (min-max) transform using (min and max) from the traning data
    """
    def __init__(self, transform_type='standardize'):
        """Params: transform_type -- str, 'standardize' or 'normalize'"""
        # initialize statistics that are calculated for each feature across all training timesteps
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

        # save setup parameters
        self.transform_type = transform_type
        self.isfit = False
        
    def fit(self, train_data):
        """
        Calculates statistics from the train data to use in standardization/normalization
        Params:
        train_data -- 2darray of shape (timesteps, features)
        """
        # use pandas so nan is ignored when calculating summary statistics
        train_data = pd.DataFrame(train_data)
        # calculate standardizing statistics
        self.mean = train_data.mean(axis=0).values.reshape(1, -1) # (1, #features)
        self.std = train_data.std(axis=0).values.reshape(1, -1) # (1, #features)
        # calculate normalizing statistics
        self.min = train_data.min(axis=0).values.reshape(1, -1) # (1, #features)
        self.max = train_data.max(axis=0).values.reshape(1, -1) # (1, #features)
        self.isfit = True
        return
    
    def transform(self, data):
        """Transforms the input data after fitting"""
        assert self.isfit == True
        if self.transform_type == 'standardize':
            result = (data - self.mean) / self.std # (timesteps, features)
        elif self.transform_type == 'normalize':
            result = (data - self.min) / (self.max - self.min) # (timesteps, features)
        else:
            raise Exception("transform_type must be 'standardize' or 'normalize'")
        return result
    
    def inverse_transform(self, data):
        """Inverses a transformation"""
        assert self.isfit == True
        if self.transform_type == 'standardize':
            result = data * self.std + self.mean # (timesteps, features)
        elif self.transform_type == 'normalize':
            result = data * (self.max - self.min) + self.min
        else:
            raise Exception("transform_type must be 'standardize' or 'normalize'")    
        return result
    
class nan_filler():
    """Class to fill na in the data (2darray of shape timesteps, features)"""
    def __init__(self, method='mean'):
        """Params: method -- str, 'mean' (fill with mean from training data) or 'value' (provide custom value to fill na)"""
        self.method = method
        # save mean from training data (if filling with mean)
        self.train_mean = None
        # custom values (if filling with custom values)\
        self.values = None
        return
    
    def fill_nan(self, data, **kwargs):
        """
        Method to fill nan
        Params:
        data -- data with nan of shape (timesteps, features)
        Optional keyword arguments:
        training_data -- data (timesteps, features). MUST PROVIDE if method = 'mean'
        values -- custom values to fill na with. MUST PROVIDE if method = 'value'
        """
        # convert data to pandas df
        data = pd.DataFrame(data)

        # read key word arguments
        if 'training_data' in kwargs:
            training_data = pd.DataFrame(kwargs['training_data'])
            self.train_mean = training_data.mean(axis=0)
        if 'values' in kwargs:
            self.values = kwargs['values']

        if self.method == 'mean':
            assert self.train_mean is not None # need to provide training data (and calculate mean) if filling with mean
            assert self.train_mean.shape[0] == data.shape[1] # training data and mean data have same number of dimensions (columns)
            return data.fillna(value=self.train_mean, inplace=False).values

        elif self.method == 'value':
            assert self.values is not None # need to provide values if filling with custom values
            return data.fillna(value=self.values, inplace=False).values
        
        else:
            raise Exception("method must be 'mean' or 'value'")


def split_and_pad(data, chunk_size=3*365, pad_value=-1, pad_nan=False):
    """
    Splits data into chunks (along time dimension) and pads them so each chunk is of the same length.
    Also pads nan values.
    Splitting a timeseries into chunks may provide more stable training. 
    Parameters:
    data -- input data, tensor of shape (timesteps, num_features)
    chunk_size -- int, number of timesteps in each chunk
    pad_value -- int, number that represents padding or nan values
    pad_nan -- bool, whether or not to also pad nan with pad value (note: nan is filled later in pipeline)
    Returns: 
    Tensor of shape (num_splits, chunk_size, num_features)
    """

    # Split the data into chunks (each chunks is of shape (chunk_size, num_features))
    data_splits = torch.split(data, chunk_size, dim=0)

    # Pad each chunk so that they are the same size
    data_new = pad_sequence(data_splits, batch_first=True, padding_value=pad_value).to(torch.float) 

    # Pad nan (default = False)
    if pad_nan:
        data_new[data_new.isnan()] = pad_value

    return data_new # output is of shape (num_splits, chunk_size, num_features)

class processing_pipeline():
    """
    Pipeline for processing data:
    1. Split data into train/val/test sets
    2. Standardize the data using the training set
    3. Fill nan using mean from train data
    4. Convert data to Pytorch tensors
    5. Split data into chunks and pad the remainder
    """
    def __init__(self, train_frac=0.6, val_frac=0.2, test_frac=0.2, chunk_size=3*365, 
                 pad_value=-1, transform_type='standardize', fill_na_method='mean'):
        """ 
        Params:
        train_frac, val_frac, test_frac -- fractions to partition train/val/test split
        chunk_size -- size of each chunk when splitting data
        pad_value -- value to pad the remainder after splitting data into chunks
        transform_type -- str, type of pre-transformation 'standardize' or 'normalize'
        fill_na_method -- str, method to fill nan 'mean' or 'values'. *CURRENTLY ONLY SUPPORTS 'mean'
        """
        # save pipeline parameters
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.chunk_size = chunk_size
        self.pad_value = pad_value
        self.transform_type = transform_type
        self.fill_na_method = fill_na_method

        # instantiate scaler object
        self.scaler = time_scaler(transform_type=self.transform_type)
        # instantiate nan filler object
        self.filler = nan_filler(method=self.fill_na_method)

    def process_data(self, data):
        """
        Process raw input data of shape (timesteps, features)
        Returns: (train, val, test) processed PyTorch tensors of shape (#chunks, chunk_size, #features)
        """
        # 1. Split into training/validation/testing sets
        df_train, df_val, df_test = train_val_test(data=data, train_frac=self.train_frac, val_frac=self.val_frac, test_frac=self.test_frac)
        df_train, df_val, df_test = df_train.astype('float'), df_val.astype('float'), df_test.astype('float') # cast data to float

        # 2. Scale the data using the training set (only for non-binary columns)
        is_binary = (df_train.isin([0, 1]) | df_train.isna()).all(axis=0)        
        nonbin_cols = df_train.columns[~is_binary]
        self.scaler.fit(df_train[nonbin_cols])
        df_train.loc[:, nonbin_cols] = self.scaler.transform(df_train[nonbin_cols])
        df_val.loc[:,   nonbin_cols] = self.scaler.transform(df_val[nonbin_cols])
        df_test.loc[:,  nonbin_cols] = self.scaler.transform(df_test[nonbin_cols])

        # self.scaler.fit(df_train)
        # df_train, df_val, df_test = (self.scaler.transform(df_train), self.scaler.transform(df_val), 
        #                              self.scaler.transform(df_test))

        # 3. Fill nan with mean from training data
        df_train, df_val, df_test = (self.filler.fill_nan(data=df_train, training_data=df_train), 
                                     self.filler.fill_nan(df_val, training_data=df_train), self.filler.fill_nan(df_test, training_data=df_train))
        
        # 4. Convert data to PyTorch tensors (from numpy array)
        ts_train, ts_val, ts_test = torch.tensor(df_train, dtype=torch.float), torch.tensor(df_val, dtype=torch.float), torch.tensor(df_test, dtype=torch.float)

        # 5. Split data into chunks and pad the remainder (for training and validation sets)
        ts_train, ts_val, ts_test = (split_and_pad(ts_train, chunk_size=self.chunk_size), 
                                     split_and_pad(ts_val, chunk_size=self.chunk_size), 
                                     split_and_pad(ts_test, chunk_size=self.chunk_size))

        return ts_train, ts_val, ts_test