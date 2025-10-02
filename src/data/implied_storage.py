### SIMPLE FUNCTION TO CALCULATE IMPLIED STORAGES ###
import pandas as pd
import numpy as np

# def get_implied_storage(inflow, outflow, initial_storage=1, initial_release=1):
#     """
#     Calculate implied storage given inflow and outflow arrays
#     """

#     implied_storages = []
#     prev_implied_storage = initial_storage
#     prev_release = initial_release
#     for i in range(len(inflow)):
#         # Calculate current implied storage
#         current_implied_storage = prev_implied_storage + inflow[i] - prev_release
#         implied_storages.append(current_implied_storage)

#         # Update prev_release and prev_implied_storage for next timestep
#         prev_implied_storage = current_implied_storage
#         prev_release = outflow[i]
#     return implied_storages

def get_implied_storage(inflow, outflow, initial_storage=0):
    """
    Calculate implied storage given inflow and outflow arrays
    """
    # initialize implied storage
    implied_storage = initial_storage
    implied_storages = [implied_storage]

    for i in range(len(inflow)):
        # Calculate next implied storage
        implied_storage = implied_storage + inflow[i] - outflow[i]
        
        if i != len(inflow) - 1:
            implied_storages.append(implied_storage)
    return implied_storages