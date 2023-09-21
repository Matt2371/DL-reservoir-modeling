### FUNCTIONS AND CLASSES TO HELP ANALYZE THE RELATIONSHIP BETWEEN LSTM CELL STATES AND OBSERVED MASS STATES ###
import numpy as np

def flatten_cells(cell_states):
    """ 
    Flattens raw model output cell states (combines chunks/batches)
    Params:
    cell_states -- PyTorch tensor, raw cell state from lstm (num_chunks, chink_size, hidden_size)
    Returns:
    flattened_cell_states -- PyTorch tensor, flattened result of shape (timesteps, hidden_size)
    """
    flattened_cell_states = cell_states.reshape(-1, cell_states.shape[-1]).numpy()
    return flattened_cell_states


def cell_correlations(cell_states, storage_states):
    """
    Calculate correlation coefficients for each series of
    cell states.
    Params:
    cell_states -- pytorch tensor, raw cell state from lstm (num_chunks, chunk_size, hidden_size)
    storage_states -- numpy 1D array, series of storage states to compare cell states with (timesteps, )
    """
    # Combine chunks and convert to numpy
    cells = flatten_cells(cell_states)
    # Flatten storage states
    storage = storage_states.flatten()

    # Loop over each cell state, and calculate correlation coef with storage
    correlations = []
    for i in range(cells.shape[-1]):
        cell_i = cells[:, i].flatten()
        # truncate cell_i to remove pads (extra timesteps are created as a result of split and padding inputs into a discrete number of chunks)
        cell_i = cell_i[:len(storage)]
        # calculate correlations - apply numpy mask to ignore nan storage
        corr = np.ma.corrcoef(np.ma.masked_invalid(storage), np.ma.masked_invalid(cell_i)).data[0, 1] # retrieve value from correlation matrix
        correlations.append(corr)
    return correlations

def plot_storage_cell(cell_states_all, storage_states, cell_id, ax, transform_type='standardize'):
    """ 
    Plot timeseries of particular cell state vs storage states.
    Params:
    cell_states_all -- pytorch tensor, raw cell state from lstm (num_chunks, chunk_size, hidden_size)
    storage_states -- numpy 1D array, series of storage states to compare cell states with (timesteps, )
    cell_id -- int, which cell unit to plot
    ax -- matplotlib axes to plot on
    transform_type -- str, 'standardize' or 'normalize.' How to scale storage/cell states
    """

    # Cell states: combine chunks and convert to numpy
    x = cell_states_all.reshape(-1, cell_states_all.shape[-1]).numpy().copy()
    # Flatten storage states
    y = storage_states.flatten().copy()

    # Select cell unit of interest
    x = x[:, cell_id]

    # Truncate to remove pads (extra timesteps are created as a result of split and padding inputs into a discrete number of chunks)
    x = x[:len(y)]


    # Scale series
    if transform_type == 'standardize':
        x = (x - x[~np.isnan(x)].mean()) / x[~np.isnan(x)].std()
        y = (y - y[~np.isnan(y)].mean()) / y[~np.isnan(y)].std()
    elif transform_type == 'normalize':
        x = (x - x[~np.isnan(x)].min()) / (x[~np.isnan(x)].max() - x[~np.isnan(x)].min())
        y = (y - y[~np.isnan(y)].min()) / (y[~np.isnan(y)].max() - y[~np.isnan(y)].min())
    else:
        raise Exception("Transform_type must be 'standardize' or 'normalize'")

    # Plot
    ax.plot(y, label=f'observed storage')
    ax.plot(x, label=f'cell state {cell_id}')
    ax.set_title(f'cell state {cell_id} vs storage')
    
    return