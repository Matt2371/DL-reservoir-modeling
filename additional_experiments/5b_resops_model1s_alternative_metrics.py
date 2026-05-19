# Evaluate saved Model 1-S ResOPS models with observed or implied storage using overall and percentile-split alternative metrics.

import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(parent_dir)

from wakepy import keep

from src.data.data_fetching import filter_res, get_left_years
from additional_experiments.alternative_metrics_utils import (
    data_processing_pytorch,
    eval_pred_train_val_test_metrics_torch,
    eval_pred_train_val_test_percentile_pbias_rmse_torch,
    eval_train_val_test_metrics,
    eval_train_val_test_percentile_pbias_rmse,
    load_resops_model1,
    metrics_dicts_to_frames,
    percentile_metric_dicts_to_frames,
    predict_sub_implied_storage,
)


############# MODEL 1-S EVALUATION HELPERS #############

def eval_one_reservoir_model1s_alternative_metrics(res_id, left_year, implied_storage=False):
    """
    Evaluate saved Model 1-S with observed or implied storage input using overall PBIAS and RMSE.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = data_processing_pytorch(
        res_id=res_id,
        transform_type='standardize',
        left=f'{left_year}-01-01',
        return_scaler=True,
        storage=True
    )

    model = load_resops_model1(res_id=res_id, storage=True)

    if implied_storage:
        conv_factor = 86400 / 1000000
        y_hat_train = predict_sub_implied_storage(model=model, x=X_train, initial_storage=0, mean=scaler.mean, std=scaler.std, conv_factor=conv_factor)[0]
        y_hat_val = predict_sub_implied_storage(model=model, x=X_val, initial_storage=0, mean=scaler.mean, std=scaler.std, conv_factor=conv_factor)[0]
        y_hat_test = predict_sub_implied_storage(model=model, x=X_test, initial_storage=0, mean=scaler.mean, std=scaler.std, conv_factor=conv_factor)[0]

        return eval_pred_train_val_test_metrics_torch(
            y_hat_train=y_hat_train,
            y_hat_val=y_hat_val,
            y_hat_test=y_hat_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            scaler=scaler,
            feature_idx=1
        )
    else:
        return eval_train_val_test_metrics(
            model=model,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            scaler=scaler,
            feature_idx=1
        )


def eval_one_reservoir_model1s_percentile_metrics(res_id, left_year, implied_storage=False, lower_quantile=0.90):
    """
    Evaluate saved Model 1-S with observed or implied storage input using percentile-split PBIAS and RMSE.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = data_processing_pytorch(
        res_id=res_id,
        transform_type='standardize',
        left=f'{left_year}-01-01',
        return_scaler=True,
        storage=True
    )

    model = load_resops_model1(res_id=res_id, storage=True)

    if implied_storage:
        conv_factor = 86400 / 1000000
        y_hat_train = predict_sub_implied_storage(model=model, x=X_train, initial_storage=0, mean=scaler.mean, std=scaler.std, conv_factor=conv_factor)[0]
        y_hat_val = predict_sub_implied_storage(model=model, x=X_val, initial_storage=0, mean=scaler.mean, std=scaler.std, conv_factor=conv_factor)[0]
        y_hat_test = predict_sub_implied_storage(model=model, x=X_test, initial_storage=0, mean=scaler.mean, std=scaler.std, conv_factor=conv_factor)[0]

        return eval_pred_train_val_test_percentile_pbias_rmse_torch(
            y_hat_train=y_hat_train,
            y_hat_val=y_hat_val,
            y_hat_test=y_hat_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            scaler=scaler,
            feature_idx=1,
            lower_quantile=lower_quantile
        )
    else:
        return eval_train_val_test_percentile_pbias_rmse(
            model=model,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            scaler=scaler,
            feature_idx=1,
            lower_quantile=lower_quantile
        )


def eval_all_reservoirs_model1s_alternative_metrics(res_list, left_years_dict, implied_storage=False):
    """
    Evaluate saved Model 1-S across all selected reservoirs.
    """
    metrics_by_res = {}
    for res in res_list:
        metrics_by_res[res] = eval_one_reservoir_model1s_alternative_metrics(
            res_id=res,
            left_year=left_years_dict[res],
            implied_storage=implied_storage
        )
    return metrics_dicts_to_frames(res_list, metrics_by_res)


def eval_all_reservoirs_model1s_percentile_metrics(res_list, left_years_dict, implied_storage=False, lower_quantile=0.90):
    """
    Evaluate saved Model 1-S across all selected reservoirs using percentile-split PBIAS and RMSE.
    """
    metrics_by_res = {}
    for res in res_list:
        metrics_by_res[res] = eval_one_reservoir_model1s_percentile_metrics(
            res_id=res,
            left_year=left_years_dict[res],
            implied_storage=implied_storage,
            lower_quantile=lower_quantile
        )
    pbias_lower_df, pbias_upper_df = percentile_metric_dicts_to_frames(res_list, metrics_by_res, metric_name='pbias')
    rmse_lower_df, rmse_upper_df = percentile_metric_dicts_to_frames(res_list, metrics_by_res, metric_name='rmse')
    return pbias_lower_df, pbias_upper_df, rmse_lower_df, rmse_upper_df


############# MAIN EXPORT SCRIPT #############

def main():
    result_dir = 'report/results/additional_experiments/alternative_metrics'
    lower_quantile = 0.90
    os.makedirs(result_dir, exist_ok=True)

    res_list = filter_res()
    left_years_dict = get_left_years(res_list=res_list)

    pbias_df, rmse_df = eval_all_reservoirs_model1s_alternative_metrics(
        res_list=res_list,
        left_years_dict=left_years_dict,
        implied_storage=False
    )
    pbias_df.to_csv(f'{result_dir}/resops_model1S_pbias.csv')
    rmse_df.to_csv(f'{result_dir}/resops_model1S_rmse.csv')
    pbias_lower_df, pbias_upper_df, rmse_lower_df, rmse_upper_df = eval_all_reservoirs_model1s_percentile_metrics(
        res_list=res_list,
        left_years_dict=left_years_dict,
        implied_storage=False,
        lower_quantile=lower_quantile
    )
    lower_pct = int(round(lower_quantile * 100))
    upper_pct = 100 - lower_pct
    pbias_lower_df.to_csv(f'{result_dir}/resops_model1S_pbias_bottom{lower_pct}.csv')
    pbias_upper_df.to_csv(f'{result_dir}/resops_model1S_pbias_top{upper_pct}.csv')
    rmse_lower_df.to_csv(f'{result_dir}/resops_model1S_rmse_bottom{lower_pct}.csv')
    rmse_upper_df.to_csv(f'{result_dir}/resops_model1S_rmse_top{upper_pct}.csv')

    pbias_df, rmse_df = eval_all_reservoirs_model1s_alternative_metrics(
        res_list=res_list,
        left_years_dict=left_years_dict,
        implied_storage=True
    )
    pbias_df.to_csv(f'{result_dir}/resops_model1S_implied_pbias.csv')
    rmse_df.to_csv(f'{result_dir}/resops_model1S_implied_rmse.csv')
    pbias_lower_df, pbias_upper_df, rmse_lower_df, rmse_upper_df = eval_all_reservoirs_model1s_percentile_metrics(
        res_list=res_list,
        left_years_dict=left_years_dict,
        implied_storage=True,
        lower_quantile=lower_quantile
    )
    pbias_lower_df.to_csv(f'{result_dir}/resops_model1S_implied_pbias_bottom{lower_pct}.csv')
    pbias_upper_df.to_csv(f'{result_dir}/resops_model1S_implied_pbias_top{upper_pct}.csv')
    rmse_lower_df.to_csv(f'{result_dir}/resops_model1S_implied_rmse_bottom{lower_pct}.csv')
    rmse_upper_df.to_csv(f'{result_dir}/resops_model1S_implied_rmse_top{upper_pct}.csv')


if __name__ == '__main__':
    with keep.running():
        main()
