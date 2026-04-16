# Evaluate saved sklearn benchmark ResOPS models with overall and percentile-split alternative metrics.

import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(parent_dir)

from src.data.data_fetching import filter_res, get_left_years
from reviewer_comments.alternative_metrics_utils import (
    data_processing_benchmark,
    eval_pred_train_val_test_metrics_numpy,
    eval_pred_train_val_test_percentile_pbias_rmse_numpy,
    load_benchmark_model,
    metrics_dicts_to_frames,
    percentile_metric_dicts_to_frames,
    unscale_standard_series,
)


############# SKLEARN BENCHMARK EVALUATION HELPERS #############

def eval_one_reservoir_benchmark_alternative_metrics(res_id, left_year, model_type):
    """
    Evaluate one saved sklearn benchmark model with overall PBIAS and RMSE.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = data_processing_benchmark(
        res_id=res_id,
        left_year=left_year,
        return_scaler=True,
        storage=True
    )

    model = load_benchmark_model(res_id=res_id, model_type=model_type)

    y_hat_train = unscale_standard_series(model.predict(X_train), scaler=scaler, feature_idx=1)
    y_hat_val = unscale_standard_series(model.predict(X_val), scaler=scaler, feature_idx=1)
    y_hat_test = unscale_standard_series(model.predict(X_test), scaler=scaler, feature_idx=1)
    y_train = unscale_standard_series(y_train, scaler=scaler, feature_idx=1)
    y_val = unscale_standard_series(y_val, scaler=scaler, feature_idx=1)
    y_test = unscale_standard_series(y_test, scaler=scaler, feature_idx=1)

    return eval_pred_train_val_test_metrics_numpy(
        y_hat_train=y_hat_train,
        y_hat_val=y_hat_val,
        y_hat_test=y_hat_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )


def eval_one_reservoir_benchmark_percentile_metrics(res_id, left_year, model_type, lower_quantile=0.90):
    """
    Evaluate one saved sklearn benchmark model with percentile-split PBIAS and RMSE.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = data_processing_benchmark(
        res_id=res_id,
        left_year=left_year,
        return_scaler=True,
        storage=True
    )

    model = load_benchmark_model(res_id=res_id, model_type=model_type)

    y_hat_train = unscale_standard_series(model.predict(X_train), scaler=scaler, feature_idx=1)
    y_hat_val = unscale_standard_series(model.predict(X_val), scaler=scaler, feature_idx=1)
    y_hat_test = unscale_standard_series(model.predict(X_test), scaler=scaler, feature_idx=1)
    y_train = unscale_standard_series(y_train, scaler=scaler, feature_idx=1)
    y_val = unscale_standard_series(y_val, scaler=scaler, feature_idx=1)
    y_test = unscale_standard_series(y_test, scaler=scaler, feature_idx=1)

    return eval_pred_train_val_test_percentile_pbias_rmse_numpy(
        y_hat_train=y_hat_train,
        y_hat_val=y_hat_val,
        y_hat_test=y_hat_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        lower_quantile=lower_quantile
    )


def eval_all_reservoirs_benchmark_alternative_metrics(res_list, left_years_dict, model_type):
    """
    Evaluate one saved sklearn benchmark model family across all selected reservoirs.
    """
    metrics_by_res = {}
    for res in res_list:
        metrics_by_res[res] = eval_one_reservoir_benchmark_alternative_metrics(
            res_id=res,
            left_year=left_years_dict[res],
            model_type=model_type
        )
    return metrics_dicts_to_frames(res_list, metrics_by_res)


def eval_all_reservoirs_benchmark_percentile_metrics(res_list, left_years_dict, model_type, lower_quantile=0.90):
    """
    Evaluate one saved sklearn benchmark model family with percentile-split PBIAS and RMSE across all selected reservoirs.
    """
    metrics_by_res = {}
    for res in res_list:
        metrics_by_res[res] = eval_one_reservoir_benchmark_percentile_metrics(
            res_id=res,
            left_year=left_years_dict[res],
            model_type=model_type,
            lower_quantile=lower_quantile
        )
    pbias_lower_df, pbias_upper_df = percentile_metric_dicts_to_frames(res_list, metrics_by_res, metric_name='pbias')
    rmse_lower_df, rmse_upper_df = percentile_metric_dicts_to_frames(res_list, metrics_by_res, metric_name='rmse')
    return pbias_lower_df, pbias_upper_df, rmse_lower_df, rmse_upper_df


############# MAIN EXPORT SCRIPT #############

def main():
    result_dir = 'report/results/reviewer_comments/alternative_metrics'
    lower_quantile = 0.90
    os.makedirs(result_dir, exist_ok=True)

    res_list = filter_res()
    left_years_dict = get_left_years(res_list=res_list)

    for model_type in ['linear', 'random_forest']:
        pbias_df, rmse_df = eval_all_reservoirs_benchmark_alternative_metrics(
            res_list=res_list,
            left_years_dict=left_years_dict,
            model_type=model_type
        )
        pbias_df.to_csv(f'{result_dir}/resops_benchmark_{model_type}_pbias.csv')
        rmse_df.to_csv(f'{result_dir}/resops_benchmark_{model_type}_rmse.csv')

        pbias_lower_df, pbias_upper_df, rmse_lower_df, rmse_upper_df = eval_all_reservoirs_benchmark_percentile_metrics(
            res_list=res_list,
            left_years_dict=left_years_dict,
            model_type=model_type,
            lower_quantile=lower_quantile
        )
        lower_pct = int(round(lower_quantile * 100))
        upper_pct = 100 - lower_pct
        pbias_lower_df.to_csv(f'{result_dir}/resops_benchmark_{model_type}_pbias_bottom{lower_pct}.csv')
        pbias_upper_df.to_csv(f'{result_dir}/resops_benchmark_{model_type}_pbias_top{upper_pct}.csv')
        rmse_lower_df.to_csv(f'{result_dir}/resops_benchmark_{model_type}_rmse_bottom{lower_pct}.csv')
        rmse_upper_df.to_csv(f'{result_dir}/resops_benchmark_{model_type}_rmse_top{upper_pct}.csv')


if __name__ == '__main__':
    main()
