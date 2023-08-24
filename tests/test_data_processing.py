
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
    
    