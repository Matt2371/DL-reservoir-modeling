import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import unittest
from src.data.data_processing import *

class test_data_processing(unittest.TestCase):
    """Test data_processing.py"""
    def test_data_split(self):
        """Testing train/val/test split"""
        input = np.ones((11, 3)) #(timesteps, features)
        expected_train = np.ones((7, 3))
        expected_val = np.ones((2, 3))
        expected_test = np.ones((2, 3))
        np.testing.assert_almost_equal(train_val_test(input)[0], expected_train)
        np.testing.assert_almost_equal(train_val_test(input)[1], expected_val)
        np.testing.assert_almost_equal(train_val_test(input)[2], expected_test)

    def test_time_standardizer1(self):
        """Testing time standardizer - transform and inverse transform"""
        input = np.array([[1, 10], [1, 12], [5, 11]])
        st_col1 = [np.array([1, 1, 5]).mean(), np.array([1, 1, 5]).std(ddof=1)]
        st_col2 = [np.array([10, 12, 11]).mean(), np.array([10, 12, 11]).std(ddof=1)]
        expected = np.hstack([(np.array([1, 1, 5]).reshape(-1, 1) - st_col1[0]) / st_col1[1],
                             (np.array([10, 12, 11]).reshape(-1, 1) - st_col2[0]) / st_col2[1]])
        
        scaler = time_standardizer()
        scaler.fit(input)
        # test transform
        np.testing.assert_almost_equal(scaler.transform(input), expected)
        # test inverse transform
        np.testing.assert_almost_equal(scaler.inverse_transform(expected), input)
    
    def test_split_and_pad1(self):
        """Testing split and pad function"""
        pad_value= -1
        input = np.ones((5, 4)) #(timesteps, features)
        # get expected tensor for chunk size of 2
        expected = np.ones((3, 2, 4)) #(num chunks, chunk size, features)
        expected[2, 1, :] = -1 # last 1 timesteps (last chunk, last step in chunk) are padded (since 5%2 = 1)
        np.testing.assert_almost_equal(split_and_pad(torch.tensor(input), chunk_size=2, 
                                                     pad_value=pad_value).numpy(), expected)
        
    def test_split_and_pad2(self):
        """Testing split and pad function -- padding nan"""
        pad_value= -1
        input = np.ones((5, 4)) #(timesteps, features)
        input[0, 0] = np.nan # add nan to input on the first timestep and first feature
        # get expected tensor for chunk size of 2
        expected = np.ones((3, 2, 4)) #(num chunks, chunk size, features)
        expected[2, 1, :] = pad_value # last 1 timesteps (last chunk, last step in chunk) are padded (since 5%2 = 1)
        expected[0, 0, 0] = pad_value # pad nan on the first timestep (first chunk, first step) and first feature
        np.testing.assert_almost_equal(split_and_pad(torch.tensor(input), chunk_size=2, 
                                                     pad_value=pad_value, pad_nan=True).numpy(), expected)
    
    def test_fill_na_mean(self):
        """Testing fill_nan function to fill na with 0"""
        input = np.array([[1.1, np.nan], [np.nan, 2.3], [np.nan, np.nan]])
        expected = np.array([[1.1, 0], [0, 2.3], [0, 0]])
        np.testing.assert_almost_equal(fill_nan(input), expected)
