import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############## READING DATA ##################

def usbr_fetch_data(name="Shasta", vars = ['inflow', 'outflow', 'storage']):
    """
    Get timeseries of selected variables for USBR reservoirs and combine into single df
    Params:
    name -- str, name of reservoir
    vars -- list of strings, variables to fetch
    """
    # store dataframes for each variables
    # start with daily timestep index (to catch missing days)
    df_list = [pd.DataFrame(index=pd.date_range('1943-12-30 08:00:00', '2023-08-23 08:00:00', freq='D'))] 
    
    # read data
    for var in vars:
        df = pd.read_csv(f'data/USBR/{name}/{var}.csv', parse_dates=True, index_col=2)[f'{var}']
        df_list.append(df)
    
    # concat data
    result = pd.concat(df_list, axis=1, join='outer')

    return result

def resops_fetch_data(res_id, vars = ['inflow', 'outflow', 'storage']):
    """
    Get timeseries of selected variables for ResOpsUS reservoirs and return single df
    Params:
    id -- int, ResOps reservoir id
    vars -- list of strings, variables to fetch
    """
    # fetch raw data
    df = pd.read_csv(f'data/ResOpsUS/time_series_all/ResOpsUS_{res_id}.csv', parse_dates=True, index_col=0)['1980-01-01':'2020-12-31']
    # select desired columns
    df = df[vars]
    return df

########## SUPPORTING FUNCTIONS ##############

def filter_res(record_frac=0.9):
    """
    Filter ResOPS reservoir ID's that have at least some percentage (default 90%) 
    of a complete inflow/outflow/storage record. Return list of reservoir ID's.
    Params:
    record_frac -- float, fraction of complete record required
    """
    # Read inflow/outflow/storage for all ResOps reservoirs
    df_inflow = pd.read_csv("data/ResOpsUS/time_series_single_variable_table/DAILY_AV_INFLOW_CUMECS.csv", parse_dates=True, index_col=0, dtype=np.float32)
    df_outflow = pd.read_csv("data/ResOpsUS/time_series_single_variable_table/DAILY_AV_OUTFLOW_CUMECS.csv", parse_dates=True, index_col=0, dtype=np.float32)
    df_storage = pd.read_csv("data/ResOpsUS/time_series_single_variable_table/DAILY_AV_STORAGE_MCM.csv", parse_dates=True, index_col=0, dtype=np.float32)

    # Columns (reservoirs) where more than 90% of record is complete for each variable
    thresh = len(df_inflow) * record_frac
    res_list = list(set(df_inflow.dropna(thresh=thresh, axis=1).columns) 
                    & set(df_outflow.dropna(thresh=thresh, axis=1).columns)
                    & set(df_storage.dropna(thresh=thresh, axis=1).columns))
    
    return res_list

def split_df_data(id, left, right=None, data_splits=(0.6, 0.2, 0.2), fill_train_mean=False):
    """
    Fetch original df data for one ResOPS or USBR reservoir in given time window, split into train/val/test portions.
    Params:
    id -- str, ResOPS reservoir ID (numeric) or USBR name (character)
    left -- str (YYYY-MM-DD), beginning boundary of time window
    right -- str (YYYY-MM-DD), end boundary of time window. If None apply default values: '2020-12-31' for ResOPS and '2022-12-31' for USBR
    data_splits -- tuple of floats, (train_frac, val_frac, test_frac) for data splits. 0.6/0.2/0.2 by default
    fill_train_mean -- whether to fill NaN with training mean (default is False)
    """
    # Read in data, columns are [inflow, outflow, storage]
    if id.isdigit():
        df = resops_fetch_data(res_id=id, vars=['inflow', 'outflow', 'storage'])
        right = '2020-12-31' if right is None else right # Default right window for ResOPS
    else:
        df = usbr_fetch_data(name=id, vars=['inflow', 'outflow', 'storage'])
        right = '2022-12-31' if right is None else right # Default right window for USBR

    # Select data window
    df = df[left:right].copy()

    assert sum(data_splits) == 1.0, "data_splits must sum to 1.0"

    # Lengths of train/val/test sets
    original_train_len = int(round(df.shape[0] * data_splits[0]))
    original_val_len = int(round(df.shape[0] * data_splits[1]))
    original_test_len = df.shape[0] - (original_train_len + original_val_len)

    # Get data, shape is (timesteps, )
    df_train = df.iloc[:original_train_len, :]
    df_val = df.iloc[original_train_len:(original_train_len+original_val_len), :]
    df_test = df.iloc[(original_train_len+original_val_len):, :]

    # (Optional) Fill NA with training mean
    if fill_train_mean:
        df_train, df_val, df_test = (df_train.fillna(df_train.mean()), df_val.fillna(df_train.mean()), df_test.fillna(df_train.mean()))

    return df_train, df_val, df_test

def get_left_years(res_list):
    """ 
    ResOPS: Get left data window years (first record year after leading NA) for each reservoir ID in res_list.
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