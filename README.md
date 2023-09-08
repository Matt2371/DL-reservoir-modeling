# DL-reservoir-modeling
Deep learning approaches to modeling reservoir releases, with a focus on interpretable LSTM's that respect mass balance. 

## Data
Early work focuses on data from Shasta Reservoir since it has a long data record between 01-01-1944 and 12-31-2022. Data includes daily
inflow, storage, and release, as well as elevation, evaporation, and precipitation. The data is sourced from the US Bureau of Reclamation.

## Methodology
1. Conduct a train/validation/test on the first 60%, 20% and 20% of the data, respectively. The data is not shuffled so that time order is preserved.
This corresponds to a period of 1944-01-01 to 1991-05-26 for the trainging set, 1991-05-27 to 2007-03-14 for the validation set, and 2007-03-15 to 2022-12-31
for the test set.
2. Train LSTM models using the training data, while validating and hyperparameter tuning on the validation set. The test set is used to evaluate final
model performance. After the models are trained, we are interested in examining how the models respect mass balance. For example, we want to explore if
an LSTM trained given inflows and the day of the year (DOY) as input will respect mass balance by learning storage states in its memory cells. 

## Source Code
The source code is organized into the following modules: data, models, and tests.

### Data
1. The data_fetching submodule reads csv data located in data/USBR/Shasta/ (NOT INCLUDED ON GITHUB) and concatenates data from the requested variales, returning
a single dataframe.
2. The data_processing submodule contains the data processing pipeline, which includes conducting a train/val/test split, standardizing the time dependent features
based on the mean and standard deviation of the training data, splitting the data into batches of 3 years long (for more stable training), and padding the remainder
with the token -1. Missing values are also filled with 0 after the data is standardized, which is equivalent to filling missing values with the mean from the training set.
The full pipeline can be accessed using the class processing_pipeline, where the process data method takes the raw 2d data (timesteps, features) and returns PyTorch tensors
of shape (batches, batch size - e.g. 3 years, features) corresponding the the train/val/test datasets.

### Models
1. The model_zoo contains class definitions for PyTorch models used in the project.
Model 1: a one layer LSTM + 1 layer FF network that takes inflow and DOY as input
Model 2: the same architecture as Model 1, except the previous predicted release is also an input to the next timestep (autoregressive LSTM)

2. The train_model submodule defines functions and classes for masking padded values from the loss function, implementing early stopping (applied by default), and conducting the training loop. The full
training loop can be accessed using the training_loop function, given the PyTorch model, optimizer, and criterion (e.g. MSE loss), and the desired patience for the early stopper.

3. The predict_model submodule contains functions and classes that support the use of models in making predictions/inferences. Notably, the flatten_rm_pad() function flattens the model output
of shape (# batches, timesteps, 1) into a 1d array, with padded values (previously to fill the remainder when splitting data into chunks) removed.

4. hyperparameter_tuning contains functions that support the hyperparameter tuning of the model. Notably, the exhaustive_grid function returns a dataframe of all possible hyperparameter combinations given
an input search space. This is useful for setting grid search.

5. analyze_lstm_cell supports the interpretation of lstm memory cells. For example, cell_correlations calculates the correlation between memory cell states and an observed state such as storage. plot_cell_storage
plots the scaled timeseries of memory cells comapred to storage states.

### Tests
Contains unittests of the experimental process throughout, with a focus on data processing and model shape in the forward pass. To run a unittest, run the command py -m unittest tests.(name of test script)

