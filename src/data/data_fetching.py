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