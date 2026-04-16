# Evaluate saved unroll LSTM / resRNN ResOPS models with overall and percentile-split alternative metrics.

import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(parent_dir)

from wakepy import keep

from src.data.data_fetching import filter_res, get_left_years
from reviewer_comments.alternative_metrics_utils import (
    data_processing_pytorch,
    eval_train_val_test_metrics,
    eval_train_val_test_percentile_pbias_rmse,
    load_resops_unroll_model,
    metrics_dicts_to_frames,
    percentile_metric_dicts_to_frames,
)


############# UNROLL LSTM / RESRNN EVALUATION HELPERS #############

def eval_one_reservoir_unroll_alternative_metrics(res_id, left_year, model_num):
    """
    Evaluate one saved unroll/resRNN model with overall PBIAS and RMSE.
    """
    train_tuple, val_tuple, test_tuple, scaler = data_processing_pytorch(
        res_id=res_id,
        transform_type='standardize',
        left=f'{left_year}-01-01',
        return_scaler=True,
        storage=False
    )

    model = load_resops_unroll_model(res_id=res_id, model_num=model_num, storage=False)

    return eval_train_val_test_metrics(
        model=model,
        X_train=train_tuple[0],
        X_val=val_tuple[0],
        X_test=test_tuple[0],
        y_train=train_tuple[1],
        y_val=val_tuple[1],
        y_test=test_tuple[1],
        scaler=scaler,
        feature_idx=1
    )


def eval_one_reservoir_unroll_percentile_metrics(res_id, left_year, model_num, lower_quantile=0.90):
    """
    Evaluate one saved unroll/resRNN model with percentile-split PBIAS and RMSE.
    """
    train_tuple, val_tuple, test_tuple, scaler = data_processing_pytorch(
        res_id=res_id,
        transform_type='standardize',
        left=f'{left_year}-01-01',
        return_scaler=True,
        storage=False
    )

    model = load_resops_unroll_model(res_id=res_id, model_num=model_num, storage=False)

    return eval_train_val_test_percentile_pbias_rmse(
        model=model,
        X_train=train_tuple[0],
        X_val=val_tuple[0],
        X_test=test_tuple[0],
        y_train=train_tuple[1],
        y_val=val_tuple[1],
        y_test=test_tuple[1],
        scaler=scaler,
        feature_idx=1,
        lower_quantile=lower_quantile
    )


def eval_all_reservoirs_unroll_alternative_metrics(res_list, left_years_dict, model_num):
    """
    Evaluate one saved unroll/resRNN model family across all selected reservoirs.
    """
    metrics_by_res = {}
    for res in res_list:
        metrics_by_res[res] = eval_one_reservoir_unroll_alternative_metrics(
            res_id=res,
            left_year=left_years_dict[res],
            model_num=model_num
        )
    return metrics_dicts_to_frames(res_list, metrics_by_res)


def eval_all_reservoirs_unroll_percentile_metrics(res_list, left_years_dict, model_num, lower_quantile=0.90):
    """
    Evaluate one saved unroll/resRNN model family with percentile-split PBIAS and RMSE across all reservoirs.
    """
    metrics_by_res = {}
    for res in res_list:
        metrics_by_res[res] = eval_one_reservoir_unroll_percentile_metrics(
            res_id=res,
            left_year=left_years_dict[res],
            model_num=model_num,
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

    for model_num in [1, 2, 3, 4]:
        pbias_df, rmse_df = eval_all_reservoirs_unroll_alternative_metrics(
            res_list=res_list,
            left_years_dict=left_years_dict,
            model_num=model_num
        )
        pbias_df.to_csv(f'{result_dir}/resops_unroll_model{model_num}_pbias.csv')
        rmse_df.to_csv(f'{result_dir}/resops_unroll_model{model_num}_rmse.csv')

        pbias_lower_df, pbias_upper_df, rmse_lower_df, rmse_upper_df = eval_all_reservoirs_unroll_percentile_metrics(
            res_list=res_list,
            left_years_dict=left_years_dict,
            model_num=model_num,
            lower_quantile=lower_quantile
        )
        lower_pct = int(round(lower_quantile * 100))
        upper_pct = 100 - lower_pct
        pbias_lower_df.to_csv(f'{result_dir}/resops_unroll_model{model_num}_pbias_bottom{lower_pct}.csv')
        pbias_upper_df.to_csv(f'{result_dir}/resops_unroll_model{model_num}_pbias_top{upper_pct}.csv')
        rmse_lower_df.to_csv(f'{result_dir}/resops_unroll_model{model_num}_rmse_bottom{lower_pct}.csv')
        rmse_upper_df.to_csv(f'{result_dir}/resops_unroll_model{model_num}_rmse_top{upper_pct}.csv')


if __name__ == '__main__':
    with keep.running():
        main()
