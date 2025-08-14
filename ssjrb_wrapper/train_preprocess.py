import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
from . import model
from scipy.optimize import differential_evolution as DE
from .util import *

##########################################################################
#   FUNCTIONS TO PROCESS TRAINING DATA TO PREPARE FOR MODEL FITTING      #
##########################################################################

""" 
Inputs:
- k, v     : node keys and variables from nodes.json
- df       : training data as dataframe
- medians  : median of each column of df by DOWY

Outputs:
- arguments into optimization algorithm (DE) for fitting reservoir/gains/pump policy
"""
############### Calculate training medians ########################
def train_medians(df):
    """
    Return median value over each DOWY for each column in df
    """
    # Initialize medians
    medians = pd.DataFrame(index=range(0,366))
    for k in df.columns:
        if k == 'dowy': 
            continue
        medians[k] = df.groupby('dowy')[k].median()

        # if any(x in k for x in ('delta', 'HRO', 'TRP')):
        #     medians[k] = df['10-01-2009':].groupby('dowy')[k].median()
        
    # Account for leap years
    medians.loc[365] = medians.loc[364] 
    return medians

################ Train/Val/Test split ############################
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

####### Functions to pull numpy arrays from dataframes ###########
def reservoir_training_data(k, v, df, medians, init_storage=False):
  dowy = df.dowy.values
  Q = df[k+'_inflow_cfs'].values
  K = v['capacity_taf'] * 1000
  R_obs = df[k+'_outflow_cfs'].values
  S_obs = df[k+'_storage_af'].values if K > 0 else np.zeros(dowy.size)
  Q_avg = medians[k+'_inflow_cfs'].values
  R_avg = medians[k+'_outflow_cfs'].values
  S_avg = medians[k+'_storage_af'].values if K > 0 else np.zeros(dowy.size)

  if not init_storage:
    S0 = S_avg[0]
  else:
    S0 = df[k+'_storage_af'].values[0]

  return (dowy, Q, K, Q_avg, R_avg, R_obs, S_avg, S_obs, S0)

def gains_training_data(df, medians):
  dowy = df.dowy.values
  Q_total = df['total_inflow_cfs'].values
  Q_total_avg = medians['total_inflow_cfs'].values
  S_total_pct = (df['total_storage'].values / 
                 np.array([medians['total_storage'].values[i] for i in df.dowy]))
  Gains_avg = medians['delta_gains_cfs'].values 
  Gains_obs = df['delta_gains_cfs'].values
  return (dowy, Q_total, Q_total_avg, S_total_pct, Gains_avg, Gains_obs)

def pump_training_data(k, v, df, medians):
  dowy = df.dowy.values
  Q_in = df.delta_inflow_cfs.values
  cap = v['capacity_cfs']
  Pump_pct_avg = medians[k+'_pumping_pct'].values
  Pump_cfs_avg = medians[k+'_pumping_cfs'].values

  if k == 'HRO':
    S_total_pct = (df['ORO_storage_af'].values / 
                   np.array([medians['ORO_storage_af'].values[i] for i in df.dowy]))
  else:
    S_total_pct = (df['total_storage'].values / 
                 np.array([medians['total_storage'].values[i] for i in df.dowy]))

  Pump_obs = df[k+'_pumping_cfs'].values
  return (dowy, Q_in, cap, Pump_pct_avg, Pump_cfs_avg, S_total_pct, Pump_obs)