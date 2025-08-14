####################################################################
# WRAPPER CLASS AROUND MODEL.PY TO SUPPORT TRAIN/VAL/TEST METHODS  #
####################################################################

from . import model # SSJRB Model definition
from .train_preprocess import * # Training prepocessing functions
from .util import * # Utility functions

# Import libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.optimize import differential_evolution as DE
from importlib.resources import files
import os

####################################################################################################################  
#                                     DEFINE SUPPORTING FUNCTIONS/CLASSES                                          #
####################################################################################################################

# Define Early Stopping callback
class EarlyStopping:
    """ 
    Implement Early Stopping callback for DE optimizer
    Params:
    objective(x, args) -- objective function, for example, model.reservoir_fit()
    val_data -- validation data to evaluate early stopping, as tuple of args to unpack into the objective function
                for example, the output of reservoir_training_data(k, v, df_val, self.medians)
    patience -- int, number of patience iterations for early stopping
    """
    def __init__(self, objective, val_data, patience):
        self.objective = objective
        self.val_data = val_data
        self.patience = patience
        self.best_score = None
        self.wait = 0
        self.best_solution = None
        self.early_stopped = False
        self.val_scores = [] # List of validation scores for each iteration
        return
    
    def __call__(self, xk, convergence):
        # Evaluate current performance on objective function
        current_score = self.objective(xk, *self.val_data)
        self.val_scores.append(current_score)

        # Monitor performance for each call/iteration
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            self.wait = 0
            self.best_solution = xk.copy()
        else:
            self.wait += 1
        
        # Trigger early stopping if patience is reached
        if self.wait >= self.patience:
            self.early_stopped = True
            raise StopIteration
        return

# Function to support model fitting
def fit_model(objective, train_data, bounds, val_data = None, patience = 10):
    """ 
    Fit model using differential evolution.
    Params:
    objective(x, args) -- objective function, for example, model.reservoir_fit()
    train_data -- training data as tuple of args to unpack into the objective function, for example, 
                  the output of reservoir_training_data(k, v, df_train, self.medians)
    bounds -- list of tuple pairs, bounds to use in optimization
    val_data -- (optional) validation data to evaluate early stopping, as tuple of args to unpack into the objective function
                for example, the output of reservoir_training_data(k, v, df_val, self.medians). If None, early stopping is not used.
    patience -- int, number of patience iterations for early stopping
    Returns:
    opt -- scipy.differential_evolution object used for optimization
    early_stopper -- early stopping callback object (return None if early stopping not used)
    """

    np.random.seed(1337)

    if val_data is None:
        opt = DE(objective, bounds=bounds, args=train_data)
        early_stopper = None
    else:
        early_stopper = EarlyStopping(objective=objective, val_data=val_data, patience=patience)
        try:
            opt = DE(objective, bounds=bounds, args=train_data, callback=early_stopper)
        except StopIteration:
            pass
    return opt, early_stopper


####################################################################################################################  
#                               SSJRB MODEL WRAPPER CLASS (SSJRB SYSTEM)                                           #
####################################################################################################################

class ssjrb_model:
    def __init__(self):
        # SSJRB model nodes
        json_path = files("ssjrb_wrapper.data") / "nodes.json"
        with json_path.open("r") as f:
            self.variables = json.load(f)
        
        # Save training parameters as dictionary
        self.params = {}
        # Save training medians (by DOWY)
        self.medians = None
        # Track training status
        self.isfit = False
        return
    
    def fit(self, df_train, df_val = None, patience = 10, cut_env_rule = False, verbose = True):
        """ 
        Train SSJRB model and save model parameters
        Params:
        df_train -- training dataframe
        df_val -- (optional) validation data to train DE with early stopping if provided
        patience -- int, number of patience iterations for early stopping
        cut_env_rule -- if True, only take training data after 10-01-2009 to reflect major env rule change
        verbose -- if True, print train/val MSE during for each node during fitting
        """
        
        # Apply env rule change cut (if applicable)
        if cut_env_rule:
            df_train = df_train['10-01-2009':]

        # Calculate training medians
        self.medians = train_medians(df_train)

        ######## Fit reservoir policy parameters ########
        for k,v in self.variables.items():
            if v['type'] != 'reservoir' or not v['fit_policy']: continue

            # Set up training parameters and bounds
            train_data = reservoir_training_data(k, v, df_train, self.medians)
            objective = model.reservoir_fit
            val_data = reservoir_training_data(k, v, df_val, self.medians) if df_val is not None else None
            bounds = [(1,3), (0,100), (100,250), (250,366), (0,1), (0,1), (0,0.2)]

            # Fit policy (use early stopping if df_val is provided)
            opt, early_stopper = fit_model(objective=objective, 
                                           train_data=train_data,
                                           val_data=val_data,
                                           bounds=bounds,
                                           patience=patience)

            # Save results
            self.params[k] = opt.x.tolist()
            if verbose:
                print((
                        f"{k}: train score={round(opt.fun)}, "
                        f"val score: {round(early_stopper.best_score) if df_val is not None else "NA"}, " 
                        f"NFE = {opt.nfev}, "
                        f"Message: {('Early stopping triggered' if early_stopper.early_stopped == True else '') if df_val is not None else ''}"
                    ))
            

        ######## Fit gains parameters #########
        # Training with no early stopping
        train_data = gains_training_data(df_train, self.medians)
        objective = model.gains_fit
        val_data = gains_training_data(df_val, self.medians) if df_val is not None else None
        bounds = [(0,2), (0.5,3), (0,1)]

        # Fit policy (use early stopping if df_val is provided)
        opt, early_stopper = fit_model(objective=objective, 
                                        train_data=train_data,
                                        val_data=val_data,
                                        bounds=bounds,
                                        patience=patience)

        # Save results
        self.params['Gains'] = opt.x.tolist()
        if verbose:
            print((
                    f"Gains: train score={round(opt.fun)}, "
                    f"val score: {round(early_stopper.best_score) if df_val is not None else "NA"}, " 
                    f"NFE = {opt.nfev}, "
                    f"Message: {('Early stopping triggered' if early_stopper.early_stopped == True else '') if df_val is not None else ''}"
                ))       

        ########  Fit pump parameters  #########
        for k,v in self.variables.items():
            if v['type'] != 'pump' or not v['fit_policy']: continue

            train_data = pump_training_data(k, v, df_train, self.medians)
            objective = model.pump_fit
            val_data = pump_training_data(k, v, df_val, self.medians) if df_val is not None else None
            bounds = [(0,5), (5000,30000), (0,5), (0,8000), (0,8000), (0,1), (0,1)]

            # Fit policy (use early stopping if df_val is provided)
            opt, early_stopper = fit_model(objective=objective, 
                                           train_data=train_data,
                                           val_data=val_data,
                                           bounds=bounds,
                                           patience=patience)

            # Save results
            self.params[k] = opt.x.tolist()
            if verbose:
                print((
                        f"{k}: train score={round(opt.fun)}, "
                        f"val score: {round(early_stopper.best_score) if df_val is not None else "NA"}, " 
                        f"NFE = {opt.nfev}, "
                        f"Message: {('Early stopping triggered' if early_stopper.early_stopped == True else '') if df_val is not None else ''}"
                    ))

        # Update training status
        self.isfit = True
        return
    
    def predict(self, df):
        """ 
        Make predictions based on input hydrology (reservoir inflow) and fitted historical medians
        Params:
        df -- dataframe containing input hydrology for each reservoir key
        Return:
        df_sim --  dataframe of simulated reservoir outflow and storage, delta gains and pumping
        """
        assert self.isfit == True

        # Load keys and fitted parameters (as tuple) and medians
        nodes = self.variables
        medians = self.medians
        params = tuple(np.array(v) for k,v in self.params.items())

        # Organize keys
        rk = [k for k in nodes.keys() if (nodes[k]['type'] == 'reservoir') and (nodes[k]['fit_policy'])]
        Kr = np.array([nodes[k]['capacity_taf'] * 1000 for k in rk])
        pk = [k for k in nodes.keys() if (nodes[k]['type'] == 'pump') and (nodes[k]['fit_policy'])]
        Kp = np.array([nodes[k]['capacity_cfs'] for k in pk])

        # Run simulation and put results in dataframe
        input_data = get_simulation_data(rk, pk, df, medians, init_storage=True)
        R,S,Delta = model.simulate(params, Kr, Kp, *input_data)

        df_sim = pd.DataFrame(index=df.index)
        df_sim['dowy'] = df.dowy

        for i,r in enumerate(rk):
            df_sim[r+'_outflow_cfs'] = R[:,i]
            df_sim[r+'_storage_af'] = S[:,i]

        delta_keys = ['delta_gains_cfs', 'delta_inflow_cfs', 'HRO_pumping_cfs', 'TRP_pumping_cfs', 'delta_outflow_cfs']
        for i,k in enumerate(delta_keys):
            df_sim[k] = Delta[:,i]
        df_sim['total_delta_pumping_cfs'] = df_sim.HRO_pumping_cfs + df_sim.TRP_pumping_cfs

        return df_sim
    
    def objectives(self, df, df_demand = None):
        """ 
        Return system objectives: reliability north and south of delta, flood volume (exceeding downstream levee capacity),
        delta peak inflow.
        Params:
        df -- dataframe containing (simulated) reservoir outflow and storage, delta gains and pumping, i.e. output of self.predict()
        df_demaind -- dataframe to account for demand multipliers
        Return:
        obj -- dataframe with resulting annual system objectives
        """
        # Load keys and fitted parameters and medians
        nodes = self.variables
        medians = self.medians

        # Get reservoir keys
        rk = [k for k in nodes.keys() if (nodes[k]['type'] == 'reservoir') and (nodes[k]['fit_policy'])]

        # Calculate objectives
        objs = results_to_annual_objectives(df, medians, nodes, rk, df_demand=df_demand)

        return objs
    
    def save_params(self, filepath):
        """
        Save fitted parameters and training medians as json.
        Params:
        filepath -- str, directory to save json files
        """
        assert self.isfit == True    
        
        # Ensure the directory exists and save files
        filepath_params = filepath + "/ssjrb_params.json"
        filepath_medians = filepath + "/ssjrb_medians.json"
        os.makedirs(os.path.dirname(filepath_params), exist_ok=True)
        os.makedirs(os.path.dirname(filepath_medians), exist_ok=True)

        # Params
        with open(filepath_params, 'w') as f:
            json.dump(self.params, f, indent=2)
        
        # Medians
        self.medians.to_json(filepath_medians, orient='columns')
        return
    
    def load_params(self, filepath):
        """ 
        Load saved parameters from json.
        Params:
        filepath -- str, directory where parameter and medians json files are saved
        """
        filepath_params = filepath + "/ssjrb_params.json"
        filepath_medians = filepath + "/ssjrb_medians.json"

        self.params = json.load(open(filepath_params))
        print("Parameters loaded successfully from json")
        self.medians = pd.read_json(filepath_medians, orient='columns')
        print("Medians loaded successfully from json")

        self.isfit = True
        return
    

####################################################################################################################  
#                      SSJRB MODEL WRAPPER CLASS (GENERIC INDIVIDUAL RESERVOIR(S))                                    #
####################################################################################################################
class reservoir_model:
    def __init__(self, reservoir_capacity):
        """
        reservoir_capacity - dictionary providing reservoir capacity (in TAF)
                             for each reservoir key. Policies will be fit for each reservoir key in the dictionary
        """
        self.reservoir_capacity = reservoir_capacity

        # Save training parameters as dictionary
        self.params = {}
        # Save training medians (by DOWY)
        self.medians = None
        # Track training status
        self.isfit = False
        return
    
    def fit(self, df_train, df_val = None, patience = 10, verbose = True):
        """ 
        Train individual reservoir(s) using SSJRB release logic and save model parameters
        Params:
        df_train -- training dataframe containing columns:
                    <reservoir key>_infow_cfs, 
                    <reservoir key>_storage_af,
                    <reservoir key>_outflow_cfs,
                    dowy (day of water year)
        df_val -- (optional) validation data to train DE with early stopping if provided.
                  Contains the same columns as df_train
        patience -- int, number of patience iterations for early stopping
        verbose -- if True, print train/val MSE during for each node during fitting
        """
        # Calculate training medians
        self.medians = train_medians(df_train)

        # Fit reservoir policy parameters
        for k, capacity in self.reservoir_capacity.items():
            # k is reservoir key
            v = {'capacity_taf': capacity}

            # Set up training parameters and bounds
            train_data = reservoir_training_data(k, v, df_train, self.medians)
            objective = model.reservoir_fit
            val_data = reservoir_training_data(k, v, df_val, self.medians) if df_val is not None else None
            bounds = [(1,3), (0,100), (100,250), (250,366), (0,1), (0,1), (0,0.2)]

            # Fit policy (use early stopping if df_val is provided)
            opt, early_stopper = fit_model(objective=objective, 
                                           train_data=train_data,
                                           val_data=val_data,
                                           bounds=bounds,
                                           patience=patience)

            # Save results
            self.params[k] = opt.x.tolist()
            if verbose:
                print((
                        f"{k}: train score={round(opt.fun)}, "
                        f"val score: {round(early_stopper.best_score) if df_val is not None else "NA"}, " 
                        f"NFE = {opt.nfev}, "
                        f"Message: {('Early stopping triggered' if early_stopper.early_stopped == True else '') if df_val is not None else ''}"
                    ))
            
        # Update training status
        self.isfit = True
        return
    
    def predict(self, df):
        """ 
        Make outflow predictions based on input hydrology (reservoir inflow) and fitted historical medians
        Params:
        df -- dataframe containing input hydrology for each reservoir key, i.e. containing columns:
                            <reservoir key>_inflow_cfs, 
                            dowy (day of water year)
        Return:
        df_sim --  dataframe of simulated reservoir outflow and storage
        """
        assert self.isfit == True

        # Load keys and fitted parameters (as tuple) and medians
        medians = self.medians
        params = tuple(np.array(v) for k,v in self.params.items())

        # Organize keys
        rk = [k for k in self.reservoir_capacity.keys()] # Reservoir list
        Kr = np.array([self.reservoir_capacity[k] * 1000 for k in rk]) # Corresponding capacities

        # Run simulation and put results in dataframe
        input_data = get_reservoir_simulation_data(rk, df, medians, init_storage=True) # dowy, Q, R_avg, S_avg, demand_multiplier, S0
        R, S = model.simulate_reservoir_only(params, Kr, *input_data)

        df_sim = pd.DataFrame(index=df.index)
        df_sim['dowy'] = df.dowy

        for i, r in enumerate(rk):
            df_sim[r+'_outflow_cfs'] = R[:, i]
            df_sim[r+'_storage_af'] = S[:, i]

        return df_sim
    
    def save_params(self, filepath, fileprefix):
        """
        Save fitted parameters and training medians as json.
        Params:
        filepath -- str, directory to save json files
        fileprefix -- str, prefix name of the file to save; _params.json and _medians.json will be appended
        """
        assert self.isfit == True    
        
        # Ensure the directory exists and save files
        filepath_params = f"{filepath}/{fileprefix}__params.json"
        filepath_medians = f"{filepath}/{fileprefix}__medians.json"
        os.makedirs(os.path.dirname(filepath_params), exist_ok=True)
        os.makedirs(os.path.dirname(filepath_medians), exist_ok=True)

        # Params
        with open(filepath_params, 'w') as f:
            json.dump(self.params, f, indent=2)
        
        # Medians
        self.medians.to_json(filepath_medians, orient='columns')
        return
    
    def load_params(self, filepath, fileprefix):
        """ 
        Load saved parameters from json.
        Params:
        filepath -- str, directory where parameter and medians json files are saved
        fileprefix -- str, prefix name of the file to load (i.e. ignoring _params.json and _medians.json)
        """
        filepath_params = f"{filepath}/{fileprefix}__params.json"
        filepath_medians = f"{filepath}/{fileprefix}__medians.json"

        self.params = json.load(open(filepath_params))
        print("Parameters loaded successfully from json")
        self.medians = pd.read_json(filepath_medians, orient='columns')
        print("Medians loaded successfully from json")

        self.isfit = True
        return