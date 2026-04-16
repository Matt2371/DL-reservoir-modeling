# Create benchmark comparison figures for overall and percentile-split alternative metrics on individual ResOps reservoirs.

import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))
sys.path.append(parent_dir)

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


RESULT_DIR = 'report/results/reviewer_comments/alternative_metrics'
LOWER_QUANTILE = 0.90
MODEL_ORDER = [
    'LSTM Model 1',
    'LSTM Model 1-S',
    'LSTM Model 2',
    'LSTM Model 3',
    'RNN Model 4',
    'Random Forest Model-S',
    'Linear Model-S',
    'Rule-Based Model',
    'Model 1-S (sim storage input)',
]


############# FIGURE HELPERS #############

def get_percentile_labels(lower_quantile):
    """
    Create readable labels for the lower and upper percentile groups.
    """
    lower_percent = int(round(lower_quantile * 100))
    upper_percent = 100 - lower_percent
    return lower_percent, upper_percent, f'bottom {lower_percent}%', f'top {upper_percent}%'


def clip_axis_to_percentiles(ax, values, quantile):
    """
    Clip extreme outliers so the boxplots remain readable.
    """
    y_lo = values.quantile(quantile)
    y_hi = values.quantile(1-quantile)
    y_pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)


def load_metric_frames(metric_name):
    """
    Load model-family metric csvs and return a single long-format dataframe.
    Params:
    metric_name -- str, one of {'pbias', 'rmse'}
    Returns:
    df_all -- concatenated long-format dataframe
    """
    metric_files = [
        ('resops_unroll_model1', 'LSTM Model 1', f'{RESULT_DIR}/resops_unroll_model1_{metric_name}.csv'),
        ('resops_model1S', 'LSTM Model 1-S', f'{RESULT_DIR}/resops_model1S_{metric_name}.csv'),
        ('resops_unroll_model2', 'LSTM Model 2', f'{RESULT_DIR}/resops_unroll_model2_{metric_name}.csv'),
        ('resops_unroll_model3', 'LSTM Model 3', f'{RESULT_DIR}/resops_unroll_model3_{metric_name}.csv'),
        ('resops_unroll_model4', 'RNN Model 4', f'{RESULT_DIR}/resops_unroll_model4_{metric_name}.csv'),
        ('resops_benchmark_random_forest', 'Random Forest Model-S', f'{RESULT_DIR}/resops_benchmark_random_forest_{metric_name}.csv'),
        ('resops_benchmark_linear', 'Linear Model-S', f'{RESULT_DIR}/resops_benchmark_linear_{metric_name}.csv'),
        ('resops_benchmark_rule_based', 'Rule-Based Model', f'{RESULT_DIR}/resops_benchmark_rule_based_{metric_name}.csv'),
        ('resops_model1S_implied', 'Model 1-S (sim storage input)', f'{RESULT_DIR}/resops_model1S_implied_{metric_name}.csv'),
    ]

    frames = []
    for _, model_label, filepath in metric_files:
        df = pd.read_csv(filepath, index_col=0)
        df = df.melt(value_vars=['train', 'val', 'test'])
        df['Model'] = model_label
        frames.append(df)

    df_all = pd.concat(frames, axis=0, ignore_index=True)
    df_all['Model'] = pd.Categorical(df_all['Model'], categories=MODEL_ORDER, ordered=True)
    return df_all


def load_percentile_metric_frames(metric_name, lower_quantile=LOWER_QUANTILE):
    """
    Load percentile-split metric csvs and return a single long-format dataframe.
    """
    lower_pct, upper_pct, lower_label, upper_label = get_percentile_labels(lower_quantile)

    metric_files = [
        ('LSTM Model 1', f'{RESULT_DIR}/resops_unroll_model1_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_unroll_model1_{metric_name}_top{upper_pct}.csv'),
        ('LSTM Model 1-S', f'{RESULT_DIR}/resops_model1S_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_model1S_{metric_name}_top{upper_pct}.csv'),
        ('LSTM Model 2', f'{RESULT_DIR}/resops_unroll_model2_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_unroll_model2_{metric_name}_top{upper_pct}.csv'),
        ('LSTM Model 3', f'{RESULT_DIR}/resops_unroll_model3_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_unroll_model3_{metric_name}_top{upper_pct}.csv'),
        ('RNN Model 4', f'{RESULT_DIR}/resops_unroll_model4_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_unroll_model4_{metric_name}_top{upper_pct}.csv'),
        ('Random Forest Model-S', f'{RESULT_DIR}/resops_benchmark_random_forest_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_benchmark_random_forest_{metric_name}_top{upper_pct}.csv'),
        ('Linear Model-S', f'{RESULT_DIR}/resops_benchmark_linear_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_benchmark_linear_{metric_name}_top{upper_pct}.csv'),
        ('Rule-Based Model', f'{RESULT_DIR}/resops_benchmark_rule_based_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_benchmark_rule_based_{metric_name}_top{upper_pct}.csv'),
        ('Model 1-S (sim storage input)', f'{RESULT_DIR}/resops_model1S_implied_{metric_name}_bottom{lower_pct}.csv', f'{RESULT_DIR}/resops_model1S_implied_{metric_name}_top{upper_pct}.csv'),
    ]

    frames = []
    for model_label, lower_path, upper_path in metric_files:
        lower_df = pd.read_csv(lower_path, index_col=0).melt(value_vars=['train', 'val', 'test'])
        lower_df['Percentile Range'] = lower_label
        lower_df['Model'] = model_label

        upper_df = pd.read_csv(upper_path, index_col=0).melt(value_vars=['train', 'val', 'test'])
        upper_df['Percentile Range'] = upper_label
        upper_df['Model'] = model_label

        frames.extend([lower_df, upper_df])

    df_all = pd.concat(frames, axis=0, ignore_index=True)
    df_all['Model'] = pd.Categorical(df_all['Model'], categories=MODEL_ORDER, ordered=True)
    return df_all


def make_metric_figure(metric_name, ylabel, filename, clip_quantile=0.02):
    """
    Create and save one boxplot figure for a given metric.
    """
    df_metric = load_metric_frames(metric_name=metric_name)

    fig, ax = plt.subplots()
    sns.boxplot(data=df_metric, x='variable', y='value', hue='Model', hue_order=MODEL_ORDER, ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=True,
        title='Model',
        fontsize='medium',
        title_fontsize='large'
    )
    legend.set_in_layout(True)

    ax.set_title(f'{metric_name.upper()} Performance on Individual ResOps Reservoirs', size='x-large')
    ax.set_ylabel(ylabel, size='x-large')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), size='x-large')
    clip_axis_to_percentiles(ax=ax, values=df_metric['value'], quantile=clip_quantile)

    plt.tight_layout()
    plt.savefig(f'{RESULT_DIR}/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_percentile_metric_figure(metric_name, ylabel, filename, lower_quantile=LOWER_QUANTILE, clip_quantile=0.02):
    """
    Create and save one boxplot figure for a percentile-split metric.
    """
    _, _, lower_label, upper_label = get_percentile_labels(lower_quantile)
    df_metric = load_percentile_metric_frames(metric_name=metric_name, lower_quantile=lower_quantile)

    fig, ax = plt.subplots()
    sns.boxplot(
        data=df_metric,
        x='Percentile Range',
        y='value',
        hue='Model',
        hue_order=MODEL_ORDER,
        order=[lower_label, upper_label],
        ax=ax
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=True,
        title='Model',
        fontsize='medium',
        title_fontsize='large'
    )
    legend.set_in_layout(True)

    ax.set_title(f'Percentile-Split {metric_name.upper()} on Individual ResOps Reservoirs', size='x-large')
    ax.set_ylabel(ylabel, size='x-large')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), size='x-large')
    clip_axis_to_percentiles(ax=ax, values=df_metric['value'], quantile=clip_quantile)

    plt.tight_layout()
    plt.savefig(f'{RESULT_DIR}/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)


############# MAIN EXPORT SCRIPT #############

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    make_metric_figure(
        metric_name='pbias',
        ylabel='PBIAS',
        filename='individual_resops_all_benchmarks_pbias.png',
        clip_quantile=0.02
    )
    make_metric_figure(
        metric_name='rmse',
        ylabel='RMSE',
        filename='individual_resops_all_benchmarks_rmse.png',
        clip_quantile=0.07
    )
    make_percentile_metric_figure(
        metric_name='pbias',
        ylabel='PBIAS',
        filename='individual_resops_all_benchmarks_percentile_pbias.png',
        lower_quantile=LOWER_QUANTILE,
        clip_quantile=0.01
    )
    make_percentile_metric_figure(
        metric_name='rmse',
        ylabel='RMSE',
        filename='individual_resops_all_benchmarks_percentile_rmse.png',
        lower_quantile=LOWER_QUANTILE,
        clip_quantile=0.05
    )


if __name__ == '__main__':
    main()
