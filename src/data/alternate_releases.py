### INCLUDE OPTION TO CALCULATE RELEASES FROM STORAGE ###
import pandas as pd
import numpy as np

def release_from_storage(df):
    """
    Calculate releases from storage: out_i = sto_i - sto_{i+1} + in_{i+1}
    Last entry will be NaN.
    Params:
    df -- dataframe of shape (timesteps, # features), inflow and storage must be columns
    """
    # Unit conversions
    cfs_to_af = 86400 / 43559.9 # sec/day * af/cf
    af_to_cfs = 1 / cfs_to_af

    # Initialize alternative releases based on storage
    alt_releases = []
    for i in range(df.shape[0]-1):
        alt_releases.append(df.storage[i] * af_to_cfs - df.storage[i+1] * af_to_cfs + df.inflow[i+1])
    alt_releases.append(float("NaN"))

    assert(len(alt_releases)) == df.shape[0]
    return pd.Series(alt_releases, index=df.index)


def get_implied_storage(inflow, outflow, initial_storage=1, initial_release=1):
    """
    Calculate implied storage given inflow and outflow arrays
    """

    implied_storages = []
    prev_implied_storage = initial_storage
    prev_release = initial_release
    for i in range(len(inflow)):
        # Calculate current implied storage
        current_implied_storage = prev_implied_storage + inflow[i] - prev_release
        implied_storages.append(current_implied_storage)

        # Update prev_release and prev_implied_storage for next timestep
        prev_implied_storage = current_implied_storage
        prev_release = outflow[i]
    return implied_storages