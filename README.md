[![DOI](https://zenodo.org/badge/656311285.svg)](https://doi.org/10.5281/zenodo.17861202)

# DL-reservoir-modeling
Diagnostics of LSTM approach to modeling large sample reservoir releases, with a focus on physical interpretability and generalization in space and time. 

## Abstract
The influence of reservoirs on the water cycle introduces significant uncertainty for hydrologic prediction. The representation of reservoirs in hydrologic models ideally must be accurate, interpretable, and transferable across sites. Recent studies have highlighted the potential for data-driven methods, including long short-term memory (LSTM) networks, to accurately capture reservoir releases. However, the performance of LSTM models of reservoir releases has not yet been diagnosed on a large-sample dataset to understand whether their accuracy is physically justified. This study evaluates the ability of LSTMs to represent reservoir release policies across the continental U.S., leveraging the recently developed ResOpsUS dataset. In particular, we focus on four key challenges to the development and application of LSTMs for this purpose: architecture selection, mass conservation, nonstationarity in time, and large-sample pooled training. We find that in many cases the LSTM succeeds in encoding physically interpretable storage dynamics in its internal states. However, the model accuracy is only weakly related to the strength of the learned storage representation; instead, it is log-linearly related to the degree of regulation. In addition, LSTMs struggle to generalize in time, where distributional shifts in operating conditions may cause unstable accuracy, and in space, where large sample pooled training fails to improve performance. This study contributes to the growing literature on interpreting deep learning models of human-hydrologic systems

## Data Source
Daily reservoir inflow, outflow, and storage data is primarily sourced for the ResOPS US dataset, with supporting experiments using data from Shasta, Folsom, Trinity, and New Melones reservoirs sourced from the US Bureau of Reclamation.

## Source Code (src/)
The source code is organized into two main modules: data and models.

### LSTM Training (src/models/)
1. model_zoo.py contains class definitions for PyTorch models used throughout the project.\
Model 1a: LSTM + 1 layer FF network that takes inflow and DOY as input. Data is standardized. Trained on MSE\
Model 1b: LSTM + 1 layer FF network that takes inflow and DOY as input. Data is normalized. Trained on MSE\
Model 1c: LSTM + 1 layer FF network that takes inflow and DOY as input. Data is normalized. Trained on RMSLE (root mean square logarithmic loss)\
Model 2: the same architecture as Model 1, except the previous predicted release is also an input to the next timestep (autoregressive LSTM)\
Model 3: LSTM + 1 layer FF. Takes inflow and DOY as input. Maintains an implied storage variable (used as input) based on previous implied storage,
previous release, and current inflow. Data is standardized. Trained on MSE.\
Model 4: RNN with implied storage. Similar to Model 3 but without LSTM gates. Trained on MSE.
2. train_model.py supports masking padded values from the loss function, implementing early stopping (applied by default), and conducting the training loop (via the training_loop function)
3. predict_model.py supports using pretrained models to make predictions/inferences. flatten_rm_pad() function flattens the model output of shape (# batches, timesteps, 1 (outflow)) into a 1d array and removes padded values.
4. hyperparameter_tuning.py supports the hyperparameter tuning of the model. exhaustive_grid() function returns a dataframe of all possible hyperparameter combinations given
an input search space.

### Data Processing (src/data/)
1. data_fetching.py: supports reading of csv data for requested variales from either ResOPS US or USBR as a pandas dataframe
2. data_processing.py details the data processing pipeline, which includes train/val/test split, standardization or normalization (min-max transform), splitting the data into Pytorch Tensors of shape (# batches, batch size, # features), padding the remainder, and filling missing values (i.e. with training mean)

## Experiments (experiments/)
Model training, exploratory analysis, and figure generation

## Tests (tests/)
Unit tests to check consistency of data processing and model shape in forward pass.



