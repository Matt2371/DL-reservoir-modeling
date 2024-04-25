import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### READING DATA ####

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