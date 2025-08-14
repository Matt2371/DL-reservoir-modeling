import numpy as np 
import matplotlib.pyplot as plt
from numba import njit
from .util import *

###########################################################
#               SSJRB MODEL DEFINITION                    #
###########################################################

'''
SSJRB simulation model. Three parts:
Reservoir releases, gains (into Delta), and Delta pumping
Each component has two functions: step() and fit()
Then all components are combined in the simulate() function
Numba compilation requires only numpy arrays, no pandas objects
'''

@njit
def get_tocs(x, d):
  tp = [0, x[1], x[2], x[3], 366]
  sp = [1, x[4], x[4], 1, 1]
  return np.interp(d, tp, sp)

@njit 
def reservoir_step(x, dowy, Q, S, K, R_avg, S_avg, tocs):
  '''
  Advances reservoir storage from one timestep to the next

    Parameters:
      x (np.array): Reservoir rule parameters (5)
      dowy (int): Day of water year
      Q (float): Inflow, cfs
      S (float): Storage, acre-feet
      K (float): Storage capacity, acre-feet
      R_avg (float): Median release for this day of the year, cfs
      S_avg (float): Median storage for this day of the year, acre-feet
      tocs (float): Top of conservation storage, fraction of capacity

    Returns:
      tuple(float, float): the updated release (cfs) and storage (af)

  '''
  R_avg *= cfs_to_afd
  Q *= cfs_to_afd

  R_target = R_avg # default

  if S < S_avg:
    R_target = R_avg * (S / S_avg) ** x[0] # exponential hedging

  S_target = S + Q - R_target # assumes 1-day forecast

  if S_target > K * tocs: # flood pool
    R_target += (S_target - K * tocs) * x[5]
    S_target = S + Q - R_target
  
  if S_target > K: # spill
    R_target += S_target - K
  elif S_target < K * x[6]: # below dead pool
    R_target -= (K * x[6] - S_target)

  R_target = np.max(np.array([R_target, 0])) 
  S_target = np.max(np.array([S + Q - R_target, 0]))

  return (R_target * afd_to_cfs, S_target)


@njit
def reservoir_fit(x, dowy, Q, K, Q_avg, R_avg, R_obs, S_avg, S_obs, S0):
  '''
  Evaluate reservoir model against historical observations for a set of parameters 

    Parameters:
      x (np.array): Reservoir rule parameters (5)
      dowy (np.array(int)): Day of water year over the simulation
      Q (np.array(float)): Inflow, cfs
      S (np.array(float)): Storage, acre-feet
      K (float): Storage capacity, acre-feet
      R_avg (np.array(float)): Median release for each day of the year, cfs
      R_obs (np.array(float)): Observed historical release, cfs
      S_avg (np.array(float)): Median storage for each day of the year, acre-feet
      S_obs (np.array(float)): Observed historical storage, acre-feet. 
        Not used currently, but could fit parameters to this instead.
      S0 (float): initial storage, acre-feet

    Returns:
      (float): the negative r**2 value of reservoir releases, to be minimized

  '''
  T = dowy.size
  R,S = np.zeros(T), np.zeros(T)
  S[0] = S0
  tocs = get_tocs(x, dowy)

  for t in range(1,T):
    inputs = (dowy[t], Q[t], S[t-1], K, R_avg[dowy[t]], S_avg[dowy[t-1]], tocs[t])
    R[t], S[t] = reservoir_step(x, *inputs)
    
  # return -np.corrcoef(R_obs, R)[0,1]**2
  return np.mean((R_obs - R) ** 2)


@njit
def gains_step(x, dowy, Q_total, Q_total_avg, S_total_pct, Gains_avg):
  '''
  Compute gains into the Delta for one timestep

    Parameters:
      x (np.array): Gains parameters 
      dowy (int): Day of water year
      Q_total (float): Total inflow to all reservoirs, cfs
      Q_total_avg (float): Average total inflow for this day of the year, cfs
      S_total_pct (float): System-wide reservoir storage, % of median
      Gains_avg (float): Median gains for this day of the year, cfs

    Returns:
      (float): Gains into the Delta for one timestep, cfs

  '''
  G = Gains_avg * S_total_pct ** x[0] # adjust up/down for wet/dry conditions
  if Q_total > x[1] * Q_total_avg: # high inflows correlated with other tributaries
    G += Q_total * x[2]
  return G


@njit
def gains_fit(x, dowy, Q_total, Q_total_avg, S_total_pct, Gains_avg, Gains_obs):
  '''
  Evaluate Delta gains model against historical observations for a set of parameters

    Parameters:
      x (np.array): Gains parameters
      dowy (np.array(int)): Day of water year
      Q_total (np.array(float)): Total inflow to all reservoirs, cfs
      Q_total_avg (np.array(float)): Average total inflow 
                                     for this day of the year, cfs
      S_total_pct (np.array(float)): System-wide reservoir storage, % of median
      Gains_avg (np.array(float)): Median gains for each day of the year, cfs
      Gains_obs (np.array(float)): Observed historical gains, cfs

    Returns:
      (float): the negative r**2 value of Delta gains, to be minimized

  '''
  T = dowy.size
  G = np.zeros(T)

  for t in range(T):
    inputs = (dowy[t], Q_total[t], Q_total_avg[dowy[t]], S_total_pct[t], Gains_avg[dowy[t]])
    G[t] = gains_step(x, *inputs)

  # return -np.corrcoef(Gains_obs, G)[0,1]**2
  return np.mean((Gains_obs - G) ** 2)


@njit
def pump_step(x, dowy, Q_in, Kp, Pump_pct_avg, Pump_avg, S_total_pct):
  '''
  Compute Delta pumping for one timestep

    Parameters:
      x (np.array): Delta pumping parameters
      dowy (int): Day of water year
      Q_in (float): Total inflow to the Delta, cfs 
                    (sum of all reservoir outflows plus gains)
      Kp (float): Pump capacity, cfs
      Pump_pct_avg (float): (Average pumping / Average inflow) 
                    for this day of the year, unitless
      S_total_pct (float): System-wide reservoir storage, % of median

    Returns:
      tuple(float, float): Pumping (cfs) and outflow (cfs)

  '''

  Q_in = np.max(np.array([Q_in, 0.0]))

  export_ratio = x[5] if dowy < 273 else x[6]
  outflow_req = x[3] if dowy < 273 else x[4]

  P = Q_in * Pump_pct_avg * S_total_pct ** x[0] # ~ export ratio

  if Q_in < x[1]: # approximate dry year adjustment
    P *= np.min(np.array([S_total_pct, 1.0])) ** x[2]

  if P > Q_in * export_ratio:
    P = Q_in * export_ratio
  if P > Q_in - outflow_req:
    P = np.max(np.array([Q_in - outflow_req, 0]))

  if 182 <= dowy <= 242: # env rule apr-may
    Kp = 750
  
  P = np.min(np.array([P, Kp]))
  return P


@njit
def pump_fit(x, dowy, Q_in, Kp, Pump_pct_avg, Pump_cfs_avg, S_total_pct, Pump_obs):
  '''
  Evaluate pump policy against historical observations for a set of parameters

    Parameters:
      x (np.array): Delta pumping parameter (1)
      dowy (np.array(int)): Day of water year
      Q_in (np.array(float)): Total inflow to the Delta, cfs 
                              (sum of all reservoir outflows plus gains)
      Kp (float): Pump capacity, cfs
      Pump_pct_avg (np.array(float)): (Average pumping / Average inflow) 
                                      for each day of the year, unitless
      S_total_pct (np.array(float)): System-wide reservoir storage, % of median
      Pump_obs (np.array(float)): Observed historical pumping, cfs

    Returns:
      (float): the negative r**2 value to be minimized

  '''
  T = dowy.size
  P = np.zeros(T)

  for t in range(T):
    inputs = (dowy[t], Q_in[t], Kp, Pump_pct_avg[dowy[t]], Pump_cfs_avg[dowy[t]], S_total_pct[t])
    P[t] = pump_step(x, *inputs)

  # return -np.corrcoef(Pump_obs, P)[0,1]**2
  return np.mean((Pump_obs - P) ** 2)


@njit
def simulate(params, Kr, Kp, dowy, Q, Q_avg, R_avg, S_avg, Gains_avg, Pump_pct_avg, Pump_avg, DM, S0):
  '''
  Run full system simulation over a given time period.

    Parameters:
      params (tuple(np.array)): Parameter arrays for all reservoirs, gains, and Delta
      Kr (np.array(float)): Reservoir capacities, af
      Kp (np.array(float)): Pump capacities, cfs
      dowy (np.array(int)): Day of water year
      Q (np.array(float, float)): Matrix of inflows at all reservoirs, cfs
      Q_avg (np.array(float, float)): Matrix of median inflows for each reservoir
                                      for each day of the year, cfs
      R_avg (np.array(float, float)): Matrix of median releases for each reservoir
                                      for each day of the year, cfs
      S_avg (np.array(float, float)): Matrix of median storage for each reservoir
                                      for each day of the year, af
      Gains_avg (np.array(float)): Median gains for each day of the year, cfs
      Pump_pct_avg (np.array(float, float)): (Median pumping / median inflow) 
                                             for each day of the year for each reservoir, unitless
      Pump_avg (np.array(float, float)): Median pumping for each day of the year
                                         for each reservoir, cfs
      DM (np.array(float)): Demand multiplier, system-wide, unitless
                            (=1.0 for the historical scenario)
      S0 (np.array(float)): Initial storage for each reservoir, af

    Returns:
      (tuple(np.array, np.array, np.array)): Matrices of timeseries results (cfs) 
        for reservoir releases, storage, and Delta Gains/Inflow/Pumping/Outflow

  '''  
  T, NR = Q.shape # timesteps, reservoirs
  NP = 2 # pumps
  R, S, G, I, P = (np.zeros((T, NR)), np.zeros((T, NR)), 
               np.zeros(T), np.zeros(T), np.zeros((T, NP)))
  Q_total = Q.sum(axis=1)
  Q_total_avg = Q_avg.sum(axis=1)
  S_total_avg = S_avg.sum(axis=1)
  S[0,:] = S0

  tocs = np.zeros((T, NR))
  for r in range(NR):
    tocs[:, r] = get_tocs(params[r], dowy)
  
  for t in range(1, T):
    d = dowy[t]

    # 1. Reservoir policies
    for r in range(NR):
      res_demand = R_avg[d,r] * DM[t] # median historical release * demand multiplier
      inputs_r = (d, Q[t, r], S[t - 1, r], Kr[r], res_demand, S_avg[d, r], tocs[t, r])
      R[t, r], S[t, r] = reservoir_step(params[r], *inputs_r)

    # 2. Gains into Delta
    S_total_pct = S[t].sum() / S_total_avg[d]
    inputs_g = (d, Q_total[t], Q_total_avg[d], 
              S_total_pct, np.min(np.array([Gains_avg[d] * DM[t], Gains_avg[d]])))
    G[t] = gains_step(params[NR], *inputs_g)

    # 3. Delta pumping policies
    I[t] = R[t].sum() + G[t]
    for p in range(NP):
      if p == 0: S_total_pct = S[t, 1] / S_avg[d, 1] # SWP - ORO only
      inputs_p = (dowy[t], I[t], Kp[p], Pump_pct_avg[d, p], Pump_avg[d, p], S_total_pct)
      P[t, p] = pump_step(params[NR + p + 1], *inputs_p)

  Delta = np.vstack((G, I, P[:, 0], P[:, 1], I - P.sum(axis=1))).T # Gains, Inflow, Pumping, Outflow
  return (R, S, Delta)


@njit
def simulate_reservoir_only(params, Kr, dowy, Q, R_avg, S_avg, DM, S0):
  '''
  Run reservoir only simulation over a given time period.

    Parameters:
      params (tuple(np.array)): Parameter arrays for all reservoirs, gains, and Delta
      Kr (np.array(float)): Reservoir capacities, af
      dowy (np.array(int)): Day of water year
      Q (np.array(float, float)): Matrix of inflows at all reservoirs, cfs
      R_avg (np.array(float, float)): Matrix of median releases for each reservoir
                                      for each day of the year, cfs
      S_avg (np.array(float, float)): Matrix of median storage for each reservoir
                                      for each day of the year, af
      DM (np.array(float)): Demand multiplier, system-wide, unitless
                            (=1.0 for the historical scenario)
      S0 (np.array(float)): Initial storage for each reservoir, af

    Returns:
      (tuple(np.array, np.array)): arrays of timeseries results (cfs) 
      for reservoir releases and storage

  '''  
  T, NR = Q.shape # timesteps, reservoirs
  R, S = (np.zeros((T, NR)), np.zeros((T, NR)))
  S[0, :] = S0

  tocs = np.zeros((T, NR))
  for r in range(NR):
    tocs[:, r] = get_tocs(params[r], dowy)
  
  for t in range(1, T):
    d = dowy[t]

    # Simulate reservoir policies
    for r in range(NR):
      res_demand = R_avg[d, r] * DM[t] # median historical release * demand multiplier
      inputs_r = (d, Q[t, r], S[t - 1, r], Kr[r], res_demand, S_avg[d, r], tocs[t, r])
      R[t, r], S[t, r] = reservoir_step(params[r], *inputs_r)

  return (R, S)