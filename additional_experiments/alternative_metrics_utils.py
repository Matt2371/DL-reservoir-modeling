# Workaround: add directory of 'src' and 'ssjrb_wrapper' to the sys.path
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(parent_dir)

import io
import contextlib
import pandas as pd
import numpy as np
import torch
from joblib import load
from sklearn.metrics import mean_squared_error

from src.data.data_processing import *
from src.data.data_processing_lagged import *
from src.data.data_fetching import *
from src.models.model_zoo import *
from src.models.predict_model import *
from ssjrb_wrapper.model_wrapper import reservoir_model
from ssjrb_wrapper.util import water_day


############# SHARED METRIC / UNSCALING HELPERS #############

def unscale_series(data, scaler, feature_idx=1):
    """
    Unscale one feature using the repo's time_scaler convention.
    """
    mean = scaler.mean[0, feature_idx]
    std = scaler.std[0, feature_idx]

    if torch.is_tensor(data):
        return data * std + mean
    else:
        data = np.asarray(data, dtype=float)
        return data * std + mean


def unscale_standard_series(data, scaler, feature_idx=1):
    """
    Unscale one feature using sklearn StandardScaler convention.
    """
    data = np.asarray(data, dtype=float)
    return data * scaler.scale_[feature_idx] + scaler.mean_[feature_idx]


def pbias(y_hat, y):
    """
    Calculate percent bias (PBIAS) between predicted and observed values.
    """
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)

    denominator = np.sum(y)
    if np.isclose(denominator, 0):
        raise ValueError("PBIAS is undefined when the sum of observed values is zero.")

    return 100 * np.sum(y - y_hat) / denominator


def rmse(y_hat, y):
    """
    Calculate root mean squared error (RMSE) between predicted and observed values.
    """
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sqrt(mean_squared_error(y_true=y, y_pred=y_hat))


def eval_pred_train_val_test_metrics_numpy(y_hat_train, y_hat_val, y_hat_test, y_train, y_val, y_test):
    """
    Calculate and return overall PBIAS and RMSE from precomputed numpy predictions.
    """
    return {
        'pbias_train': pbias(y_hat_train, y_train),
        'pbias_val': pbias(y_hat_val, y_val),
        'pbias_test': pbias(y_hat_test, y_test),
        'rmse_train': rmse(y_hat_train, y_train),
        'rmse_val': rmse(y_hat_val, y_val),
        'rmse_test': rmse(y_hat_test, y_test),
    }


def get_percentile_masks(y, lower_quantile=0.90):
    """
    Split observations into lower and upper groups using observed-value quantiles.
    Params:
    y -- observed array
    lower_quantile -- float in (0, 1), fraction of observations in the lower group
                      e.g. 0.90 gives bottom 90% vs top 10%
    Returns:
    lower_mask, upper_mask -- boolean masks for lower and upper percentile groups
    """
    y = np.asarray(y, dtype=float)
    threshold = np.quantile(y, lower_quantile)
    lower_mask = y <= threshold
    upper_mask = y > threshold
    return lower_mask, upper_mask


def percentile_pbias_rmse(y_hat, y, lower_quantile=0.90):
    """
    Evaluate PBIAS and RMSE separately on the lower and upper tails defined by observed-value quantiles.
    """
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    lower_mask, upper_mask = get_percentile_masks(y=y, lower_quantile=lower_quantile)

    def subset_pbias(mask):
        if mask.sum() == 0:
            return np.nan
        try:
            return pbias(y_hat[mask], y[mask])
        except ValueError:
            return np.nan

    lower_pbias = subset_pbias(lower_mask)
    upper_pbias = subset_pbias(upper_mask)
    lower_rmse = np.nan if lower_mask.sum() == 0 else rmse(y_hat[lower_mask], y[lower_mask])
    upper_rmse = np.nan if upper_mask.sum() == 0 else rmse(y_hat[upper_mask], y[upper_mask])
    return lower_pbias, upper_pbias, lower_rmse, upper_rmse


def eval_pred_train_val_test_percentile_pbias_rmse_numpy(y_hat_train, y_hat_val, y_hat_test, y_train, y_val, y_test, lower_quantile=0.90):
    """
    Calculate and return percentile-split PBIAS and RMSE from precomputed numpy predictions.
    """
    train_low_pbias, train_high_pbias, train_low_rmse, train_high_rmse = percentile_pbias_rmse(
        y_hat_train, y_train, lower_quantile=lower_quantile
    )
    val_low_pbias, val_high_pbias, val_low_rmse, val_high_rmse = percentile_pbias_rmse(
        y_hat_val, y_val, lower_quantile=lower_quantile
    )
    test_low_pbias, test_high_pbias, test_low_rmse, test_high_rmse = percentile_pbias_rmse(
        y_hat_test, y_test, lower_quantile=lower_quantile
    )

    return {
        'pbias_lower_train': train_low_pbias,
        'pbias_lower_val': val_low_pbias,
        'pbias_lower_test': test_low_pbias,
        'pbias_upper_train': train_high_pbias,
        'pbias_upper_val': val_high_pbias,
        'pbias_upper_test': test_high_pbias,
        'rmse_lower_train': train_low_rmse,
        'rmse_lower_val': val_low_rmse,
        'rmse_lower_test': test_low_rmse,
        'rmse_upper_train': train_high_rmse,
        'rmse_upper_val': val_high_rmse,
        'rmse_upper_test': test_high_rmse,
    }


def eval_train_val_test_metrics(model, X_train, X_val, X_test, y_train, y_val, y_test, scaler=None, feature_idx=1):
    """
    Calculate and return overall PBIAS and RMSE metrics for train/val/test sets from a PyTorch model.
    """
    y_hat_train, y_hat_val, y_hat_test = predict(model, X_train), predict(model, X_val), predict(model, X_test)
    return eval_pred_train_val_test_metrics_torch(
        y_hat_train=y_hat_train,
        y_hat_val=y_hat_val,
        y_hat_test=y_hat_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        feature_idx=feature_idx
    )


def eval_train_val_test_percentile_pbias_rmse(model, X_train, X_val, X_test, y_train, y_val, y_test, scaler=None, feature_idx=1, lower_quantile=0.90):
    """
    Calculate and return percentile-split PBIAS and RMSE metrics for train/val/test sets from a PyTorch model.
    """
    y_hat_train, y_hat_val, y_hat_test = predict(model, X_train), predict(model, X_val), predict(model, X_test)
    return eval_pred_train_val_test_percentile_pbias_rmse_torch(
        y_hat_train=y_hat_train,
        y_hat_val=y_hat_val,
        y_hat_test=y_hat_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        feature_idx=feature_idx,
        lower_quantile=lower_quantile
    )


def eval_pred_train_val_test_metrics_torch(y_hat_train, y_hat_val, y_hat_test, y_train, y_val, y_test, scaler=None, feature_idx=1):
    """
    Calculate and return overall PBIAS and RMSE from precomputed torch predictions.
    """
    y_hat_train, y_train = flatten_rm_pad(y_hat=y_hat_train, y=y_train)
    y_hat_val, y_val = flatten_rm_pad(y_hat=y_hat_val, y=y_val)
    y_hat_test, y_test = flatten_rm_pad(y_hat=y_hat_test, y=y_test)

    if scaler is not None:
        y_hat_train = unscale_series(y_hat_train, scaler=scaler, feature_idx=feature_idx)
        y_hat_val = unscale_series(y_hat_val, scaler=scaler, feature_idx=feature_idx)
        y_hat_test = unscale_series(y_hat_test, scaler=scaler, feature_idx=feature_idx)
        y_train = unscale_series(y_train, scaler=scaler, feature_idx=feature_idx)
        y_val = unscale_series(y_val, scaler=scaler, feature_idx=feature_idx)
        y_test = unscale_series(y_test, scaler=scaler, feature_idx=feature_idx)

    return eval_pred_train_val_test_metrics_numpy(
        y_hat_train=y_hat_train.detach().cpu().numpy(),
        y_hat_val=y_hat_val.detach().cpu().numpy(),
        y_hat_test=y_hat_test.detach().cpu().numpy(),
        y_train=y_train.detach().cpu().numpy(),
        y_val=y_val.detach().cpu().numpy(),
        y_test=y_test.detach().cpu().numpy()
    )


def eval_pred_train_val_test_percentile_pbias_rmse_torch(y_hat_train, y_hat_val, y_hat_test, y_train, y_val, y_test, scaler=None, feature_idx=1, lower_quantile=0.90):
    """
    Calculate and return percentile-split PBIAS and RMSE from precomputed torch predictions.
    """
    y_hat_train, y_train = flatten_rm_pad(y_hat=y_hat_train, y=y_train)
    y_hat_val, y_val = flatten_rm_pad(y_hat=y_hat_val, y=y_val)
    y_hat_test, y_test = flatten_rm_pad(y_hat=y_hat_test, y=y_test)

    if scaler is not None:
        y_hat_train = unscale_series(y_hat_train, scaler=scaler, feature_idx=feature_idx)
        y_hat_val = unscale_series(y_hat_val, scaler=scaler, feature_idx=feature_idx)
        y_hat_test = unscale_series(y_hat_test, scaler=scaler, feature_idx=feature_idx)
        y_train = unscale_series(y_train, scaler=scaler, feature_idx=feature_idx)
        y_val = unscale_series(y_val, scaler=scaler, feature_idx=feature_idx)
        y_test = unscale_series(y_test, scaler=scaler, feature_idx=feature_idx)

    return eval_pred_train_val_test_percentile_pbias_rmse_numpy(
        y_hat_train=y_hat_train.detach().cpu().numpy(),
        y_hat_val=y_hat_val.detach().cpu().numpy(),
        y_hat_test=y_hat_test.detach().cpu().numpy(),
        y_train=y_train.detach().cpu().numpy(),
        y_val=y_val.detach().cpu().numpy(),
        y_test=y_test.detach().cpu().numpy(),
        lower_quantile=lower_quantile
    )


############# PYTORCH RESOPS DATA PROCESSING / MODEL LOADERS #############

def data_processing_pytorch(res_id, transform_type, left, right='2020-12-31', train_frac=0.6, val_frac=0.2, test_frac=0.2, return_scaler=False, storage=False):
    """
    Run timeseries tensor data processing for one ResOPS reservoir.
    """
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
    df['doy'] = df.index.to_series().dt.dayofyear
    df = df[left:right].copy()

    pipeline = processing_pipeline(
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        chunk_size=3 * 365,
        pad_value=-1,
        transform_type=transform_type,
        fill_na_method='mean'
    )
    ts_train, ts_val, ts_test = pipeline.process_data(df)

    if storage:
        X_train, X_val, X_test = ts_train[:, :, [0, 2, 3]], ts_val[:, :, [0, 2, 3]], ts_test[:, :, [0, 2, 3]]
    else:
        X_train, X_val, X_test = ts_train[:, :, [0, 3]], ts_val[:, :, [0, 3]], ts_test[:, :, [0, 3]]
    y_train, y_val, y_test = ts_train[:, :, [1]], ts_val[:, :, [1]], ts_test[:, :, [1]]

    if return_scaler:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), pipeline.scaler
    else:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_resops_unroll_model(res_id, model_num, storage=False):
    """
    Load saved unroll model checkpoint from experiments/10f_resops_train_unroll.py.
    """
    input_size = 3 if storage else 2

    if model_num == 1:
        model = LSTMModel1(input_size=input_size, hidden_size1=30, hidden_size2=15,
                           output_size=1, num_layers=1, dropout_prob=0.3)
    elif model_num == 2:
        model = LSTMModel2(input_size=input_size, hidden_size1=35, hidden_size2=20,
                           num_layers=1, output_size=1, dropout_prob=0.3, initial_output=0)
    elif model_num == 3:
        model = LSTMModel3(input_size=input_size, hidden_size1=35, hidden_size2=15,
                           output_size=1, num_layers=1, dropout_prob=0.3,
                           initial_output=0, initial_implied_storage=0)
    elif model_num == 4:
        model = resRNN(input_size=input_size, hidden_size=50, output_size=1, dropout_prob=0.7)
    else:
        raise ValueError("model_num must be 1, 2, 3, or 4.")

    model_path = f'src/models/saved_models/resops_unroll_models/resops_model{model_num}_{res_id}.pt'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model


def load_resops_model1(res_id, storage=False):
    """
    Load saved Model 1 or Model 1-S checkpoint from experiments/10a_resops_training.py.
    """
    input_size = 3 if storage else 2
    model = LSTMModel1_opt(input_size=input_size, hidden_size1=30, hidden_size2=15,
                           output_size=1, num_layers=1, dropout_prob=0.3)
    model_path = f'src/models/saved_models/resops_model1/resops_model{"1S" if storage else "1"}_{res_id}.pt'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model


def predict_sub_implied_storage(model, x, initial_storage, mean, std, conv_factor):
    """
    Make predictions with implied storage replacing observed storage input.
    """
    X = x.clone()

    if initial_storage == 0:
        initial_storage = torch.zeros((X.shape[0], 1), device=X.device)
    assert initial_storage.shape == (X.shape[0], 1)

    implied_storage = (initial_storage * std[0, 2]) + mean[0, 2]
    implied_storage_list = [initial_storage]
    prediction_list = []

    for i in range(X.shape[1]):
        scaled_implied_storage = (implied_storage - mean[0, 2]) / std[0, 2]
        X[:, i, [1]] = scaled_implied_storage

        scaled_out = predict(model=model, x=X[:, i:i+1, :])
        prediction_list.append(scaled_out)

        unscaled_out = (scaled_out.squeeze(dim=-1) * std[0, 1]) + mean[0, 1]
        unscaled_in = (X[:, i, [0]] * std[0, 0]) + mean[0, 0]
        implied_storage = implied_storage + (unscaled_in - unscaled_out) * conv_factor

        if i != X.shape[1] - 1:
            implied_storage_list.append((implied_storage - mean[0, 2]) / std[0, 2])

    out = torch.cat(prediction_list, dim=1)
    implied_storages = torch.stack(implied_storage_list, dim=1)
    return out, implied_storages


############# SKLEARN BENCHMARK DATA / MODEL LOADERS #############

def data_processing_benchmark(res_id, left_year, right_year=2020, train_frac=0.6, val_frac=0.2, test_frac=0.2, return_scaler=False, storage=True):
    """
    Run lagged benchmark-model data processing for one ResOPS reservoir.
    """
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
    df['doy'] = df.index.to_series().dt.dayofyear
    df = df[f'{left_year}-01-01':f'{right_year}-12-31'].copy()

    pipeline = processing_pipeline_w_lags(
        df=df,
        n_lags=5,
        exclude_list=['outflow', 'storage', 'doy'],
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        left_year=left_year,
        right_year=right_year
    )
    df_train, df_val, df_test = pipeline.process_data()

    if storage:
        X_train, X_val, X_test = df_train[:, np.r_[0, 2:9]], df_val[:, np.r_[0, 2:9]], df_test[:, np.r_[0, 2:9]]
    else:
        X_train, X_val, X_test = df_train[:, np.r_[0, 3:9]], df_val[:, np.r_[0, 3:9]], df_test[:, np.r_[0, 3:9]]
    y_train, y_val, y_test = df_train[:, 1], df_val[:, 1], df_test[:, 1]

    if return_scaler:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), pipeline.scaler
    else:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_benchmark_model(res_id, model_type):
    """
    Load saved sklearn benchmark model.
    """
    return load(f'src/models/saved_models/resops_benchmark_models/resops_{model_type}_{res_id}.joblib')


############# RULE-BASED BENCHMARK DATA / MODEL LOADERS #############

def data_processing_rule_based(res_id, left, right='2020-12-31'):
    """
    Run rule-based benchmark data preprocessing for one ResOPS reservoir.
    """
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
    df = df[left:right].copy()

    df[f'{res_id}_inflow_cfs'] = df['inflow'] * 35.3147
    df[f'{res_id}_outflow_cfs'] = df['outflow'] * 35.3147
    df[f'{res_id}_storage_af'] = df['storage'] * 810.713

    doy_series = df.index.to_series().dt.dayofyear
    df['dowy'] = [water_day(i) for i in doy_series]

    df_train, df_val, df_test = train_val_test(df, train_frac=0.6, val_frac=0.2, test_frac=0.2)

    train_mean = df_train.mean()
    df_train = df_train.fillna(train_mean)
    df_val = df_val.fillna(train_mean)
    df_test = df_test.fillna(train_mean)

    return df_train, df_val, df_test


def load_rule_based_model(res_id, df_train):
    """
    Load saved SSJRB rule-based model parameters for one reservoir.
    """
    training_capacity = df_train[f'{res_id}_storage_af'].max() / 1000
    model = reservoir_model(reservoir_capacity={res_id: training_capacity})
    with contextlib.redirect_stdout(io.StringIO()):
        model.load_params(filepath='src/models/saved_models/resops_rule_models', fileprefix=f'resops_ssjrb_model_{res_id}')
    return model


############# CSV EXPORT HELPER #############

def metrics_dicts_to_frames(res_list, metrics_by_res):
    """
    Convert reservoir metrics dicts to pbias/rmse dataframes.
    """
    pbias_df = pd.DataFrame(index=res_list, columns=['train', 'val', 'test'], dtype=float)
    rmse_df = pd.DataFrame(index=res_list, columns=['train', 'val', 'test'], dtype=float)

    for res in res_list:
        metrics = metrics_by_res[res]
        pbias_df.loc[res, :] = (metrics['pbias_train'], metrics['pbias_val'], metrics['pbias_test'])
        rmse_df.loc[res, :] = (metrics['rmse_train'], metrics['rmse_val'], metrics['rmse_test'])

    return pbias_df, rmse_df


def percentile_metric_dicts_to_frames(res_list, metrics_by_res, metric_name):
    """
    Convert reservoir percentile metric dicts to paired lower/upper dataframes.
    """
    lower_df = pd.DataFrame(index=res_list, columns=['train', 'val', 'test'], dtype=float)
    upper_df = pd.DataFrame(index=res_list, columns=['train', 'val', 'test'], dtype=float)

    for res in res_list:
        metrics = metrics_by_res[res]
        lower_df.loc[res, :] = (
            metrics[f'{metric_name}_lower_train'],
            metrics[f'{metric_name}_lower_val'],
            metrics[f'{metric_name}_lower_test'],
        )
        upper_df.loc[res, :] = (
            metrics[f'{metric_name}_upper_train'],
            metrics[f'{metric_name}_upper_val'],
            metrics[f'{metric_name}_upper_test'],
        )

    return lower_df, upper_df
