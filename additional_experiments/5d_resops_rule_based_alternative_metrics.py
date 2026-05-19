# Evaluate saved rule-based ResOPS benchmark models with overall and percentile-split alternative metrics.

import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(parent_dir)

from src.data.data_fetching import filter_res, get_left_years
from additional_experiments.alternative_metrics_utils import (
    eval_pred_train_val_test_metrics_numpy,
    eval_pred_train_val_test_percentile_pbias_rmse_numpy,
    data_processing_rule_based,
    load_rule_based_model,
    metrics_dicts_to_frames,
    percentile_metric_dicts_to_frames,
)


############# RULE-BASED EVALUATION HELPERS #############

CFS_PER_CMS = 35.3147

def eval_one_reservoir_rule_based_alternative_metrics(res_id, left_year):
    """
    Evaluate one saved SSJRB rule-based model with overall PBIAS and RMSE.
    """
    df_train, df_val, df_test = data_processing_rule_based(
        res_id=res_id,
        left=f'{left_year}-01-01',
        right='2020-12-31'
    )

    model = load_rule_based_model(res_id=res_id, df_train=df_train)

    df_hat_train = model.predict(df=df_train)
    df_hat_val = model.predict(df=df_val)
    df_hat_test = model.predict(df=df_test)

    return eval_pred_train_val_test_metrics_numpy(
        y_hat_train=df_hat_train[f'{res_id}_outflow_cfs'].to_numpy() / CFS_PER_CMS,
        y_hat_val=df_hat_val[f'{res_id}_outflow_cfs'].to_numpy() / CFS_PER_CMS,
        y_hat_test=df_hat_test[f'{res_id}_outflow_cfs'].to_numpy() / CFS_PER_CMS,
        y_train=df_train['outflow'].to_numpy(),
        y_val=df_val['outflow'].to_numpy(),
        y_test=df_test['outflow'].to_numpy()
    )


def eval_one_reservoir_rule_based_percentile_metrics(res_id, left_year, lower_quantile=0.90):
    """
    Evaluate one saved SSJRB rule-based model with percentile-split PBIAS and RMSE.
    """
    df_train, df_val, df_test = data_processing_rule_based(
        res_id=res_id,
        left=f'{left_year}-01-01',
        right='2020-12-31'
    )

    model = load_rule_based_model(res_id=res_id, df_train=df_train)

    df_hat_train = model.predict(df=df_train)
    df_hat_val = model.predict(df=df_val)
    df_hat_test = model.predict(df=df_test)

    return eval_pred_train_val_test_percentile_pbias_rmse_numpy(
        y_hat_train=df_hat_train[f'{res_id}_outflow_cfs'].to_numpy() / CFS_PER_CMS,
        y_hat_val=df_hat_val[f'{res_id}_outflow_cfs'].to_numpy() / CFS_PER_CMS,
        y_hat_test=df_hat_test[f'{res_id}_outflow_cfs'].to_numpy() / CFS_PER_CMS,
        y_train=df_train['outflow'].to_numpy(),
        y_val=df_val['outflow'].to_numpy(),
        y_test=df_test['outflow'].to_numpy(),
        lower_quantile=lower_quantile
    )


def eval_all_reservoirs_rule_based_alternative_metrics(res_list, left_years_dict):
    """
    Evaluate saved SSJRB rule-based models across all selected reservoirs.
    """
    metrics_by_res = {}
    for res in res_list:
        metrics_by_res[res] = eval_one_reservoir_rule_based_alternative_metrics(
            res_id=res,
            left_year=left_years_dict[res]
        )
    return metrics_dicts_to_frames(res_list, metrics_by_res)


def eval_all_reservoirs_rule_based_percentile_metrics(res_list, left_years_dict, lower_quantile=0.90):
    """
    Evaluate saved SSJRB rule-based models across all selected reservoirs with percentile-split PBIAS and RMSE.
    """
    metrics_by_res = {}
    for res in res_list:
        metrics_by_res[res] = eval_one_reservoir_rule_based_percentile_metrics(
            res_id=res,
            left_year=left_years_dict[res],
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

    pbias_df, rmse_df = eval_all_reservoirs_rule_based_alternative_metrics(
        res_list=res_list,
        left_years_dict=left_years_dict
    )
    pbias_df.to_csv(f'{result_dir}/resops_benchmark_rule_based_pbias.csv')
    rmse_df.to_csv(f'{result_dir}/resops_benchmark_rule_based_rmse.csv')

    pbias_lower_df, pbias_upper_df, rmse_lower_df, rmse_upper_df = eval_all_reservoirs_rule_based_percentile_metrics(
        res_list=res_list,
        left_years_dict=left_years_dict,
        lower_quantile=lower_quantile
    )
    lower_pct = int(round(lower_quantile * 100))
    upper_pct = 100 - lower_pct
    pbias_lower_df.to_csv(f'{result_dir}/resops_benchmark_rule_based_pbias_bottom{lower_pct}.csv')
    pbias_upper_df.to_csv(f'{result_dir}/resops_benchmark_rule_based_pbias_top{upper_pct}.csv')
    rmse_lower_df.to_csv(f'{result_dir}/resops_benchmark_rule_based_rmse_bottom{lower_pct}.csv')
    rmse_upper_df.to_csv(f'{result_dir}/resops_benchmark_rule_based_rmse_top{upper_pct}.csv')


if __name__ == '__main__':
    main()
