# Permutation feature importance for pooled ResOPS model
# Group one-hot attributes into interpretable feature groups

# Workaround: add directory of 'src' and 'ssjrb_wrapper' to the sys.path
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))  # One level up to the project root
sys.path.append(parent_dir)

import copy
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.data_fetching import *
from src.data.data_processing import *
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import get_device


OUTPUT_DIR = 'report/results/reviewer_comments/hyperparameter_tuning_pooled'
CHECKPOINT_PATH = f'{OUTPUT_DIR}/resops_simul_model.pt'
OOS_RESULTS_PATH = f'{OUTPUT_DIR}/resops_oos_out_of_sample_test.csv'
SUMMARY_PATH = f'{OUTPUT_DIR}/resops_pooled_feature_importance_summary.csv'
FIGURE_PATH = f'{OUTPUT_DIR}/resops_pooled_feature_importance.png'


def get_attributes(index_type=str):
    """
    Get reservoir attributes df (categorical attributes one-hot encoded; continuous attributes kept as floats)
    Params:
    index_type -- type of index for returned dataframe, either str or int
    """
    # GRanD Attributes
    gdf = gpd.read_file("data/GRanD/GRanD_dams_v1_3.shp")
    gdf = gdf.drop(columns="geometry").set_index("GRAND_ID")

    # Main reservoir use
    use_ohe = pd.get_dummies(gdf['MAIN_USE'], prefix='USE', dtype='float')
    use_ohe.index = use_ohe.index.astype(str)

    # Capacity (standardize once across reservoirs; append later after time-series preprocessing)
    capacity = gdf['CAP_MCM'].copy()
    capacity.index = capacity.index.astype(str)
    capacity = capacity.fillna(capacity.mean())
    capacity = (capacity - capacity.mean()) / capacity.std()

    # Operating agency (one-hot encode)
    operating_agency = pd.read_csv("data/ResOpsUS/attributes/reservoir_attributes.csv", index_col=0)["AGENCY_CODE"]
    operating_agency_ohe = pd.get_dummies(operating_agency, prefix='AGENCY', dtype='float')
    operating_agency_ohe.index = operating_agency_ohe.index.astype(str)

    # DOR category (based on log(mean inflow / max storage))
    df_inflow = pd.read_csv(
        "data/ResOpsUS/time_series_single_variable_table/DAILY_AV_INFLOW_CUMECS.csv",
        parse_dates=True,
        index_col=0,
        dtype=np.float32,
    )
    df_storage = pd.read_csv(
        "data/ResOpsUS/time_series_single_variable_table/DAILY_AV_STORAGE_MCM.csv",
        parse_dates=True,
        index_col=0,
        dtype=np.float32,
    )
    df_result = pd.concat([df_inflow.mean(skipna=True), df_storage.max()], axis=1, join='inner')
    df_result.columns = ['mean_inflow', 'max_storage']

    ratio = df_result['mean_inflow'] / df_result['max_storage']
    df_result['log_mean_inflow_max_storage'] = np.where(ratio > 0, np.log(ratio), np.nan)
    df_result['log_mean_inflow_max_storage_cat'] = pd.cut(
        df_result['log_mean_inflow_max_storage'],
        bins=[-np.inf, -3.79, -3.17, -2.46, np.inf],
        labels=['very_high', 'high', 'medium', 'low'],
    )
    dor_ohe = pd.get_dummies(df_result['log_mean_inflow_max_storage_cat'], prefix='DOR', dtype='float')
    dor_ohe.index = dor_ohe.index.astype(str)

    attribute_df = use_ohe.join([dor_ohe, capacity, operating_agency_ohe], how='left')
    dummy_cols = [c for c in attribute_df.columns if c != 'CAP_MCM']
    attribute_df[dummy_cols] = attribute_df[dummy_cols].fillna(0)
    attribute_df.index = attribute_df.index.astype(index_type)
    return attribute_df


def get_input_feature_columns(df, storage=False):
    """
    Build input feature column names in the same style as experiment reviewer_comments/2a.
    Returns:
    feature_names -- ordered list of model input feature names
    processed_input_cols -- feature columns sent through processing_pipeline
    static_continuous_cols -- continuous static features appended after processing
    """
    base_input_cols = ['inflow', 'doy'] + (['storage'] if storage else [])
    attribute_cols = [c for c in df.columns if c not in base_input_cols + ['storage', 'outflow']]
    static_continuous_cols = [c for c in ['CAP_MCM'] if c in attribute_cols]
    processed_attr_cols = [c for c in attribute_cols if c not in static_continuous_cols]
    processed_input_cols = base_input_cols + processed_attr_cols
    feature_names = processed_input_cols + static_continuous_cols
    return feature_names, processed_input_cols, static_continuous_cols


def data_processing(
    res_id,
    transform_type,
    left,
    right='2020-12-31',
    train_frac=0.6,
    val_frac=0.2,
    test_frac=0.2,
    return_scaler=False,
    storage=False,
    attributes=None,
    return_feature_names=False,
):
    """
    Run data processing pipeline for one ResOPS reservoir.
    Params:
    res_id -- int, ResOPS reservoir ID
    transform_type -- str, in preprocessing, whether to 'standardize' or 'normalize' the data
    left -- str (YYYY-MM-DD), beginning boundary of time window
    right -- str (YYYY-MM-DD), end boundary of time window
    return_scaler -- bool, whether or not to return src.data.data_processing.time_scaler() object
    storage -- bool, whether or not to include storage data in features
    attributes -- pd.DataFrame, dataframe of reservoir attributes to include as features
    return_feature_names -- bool, whether or not to return ordered input feature names
    """
    # Read in data, columns are [inflow, outflow, storage]
    df = resops_fetch_data(res_id=res_id, vars=['inflow', 'outflow', 'storage'])
    df['doy'] = df.index.to_series().dt.dayofyear.astype('float')

    if attributes is not None:
        attr = attributes.loc[[res_id]]
        attr = pd.concat([attr] * len(df), ignore_index=True)
        attr.index = df.index
        df = pd.concat([df, attr], axis=1)

    df = df[left:right].copy()

    feature_names, processed_input_cols, static_continuous_cols = get_input_feature_columns(df=df, storage=storage)
    df_processed = df[processed_input_cols + ['outflow']].copy()
    target_idx = [len(processed_input_cols)]

    pipeline = processing_pipeline(
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        chunk_size=3 * 365,
        pad_value=-1,
        transform_type=transform_type,
        fill_na_method='mean',
    )
    ts_train, ts_val, ts_test = pipeline.process_data(df_processed)

    X_train = ts_train[:, :, :len(processed_input_cols)]
    X_val = ts_val[:, :, :len(processed_input_cols)]
    X_test = ts_test[:, :, :len(processed_input_cols)]
    y_train = ts_train[:, :, target_idx]
    y_val = ts_val[:, :, target_idx]
    y_test = ts_test[:, :, target_idx]

    if static_continuous_cols:
        static_attr = torch.tensor(df[static_continuous_cols].iloc[0].values, dtype=torch.float).view(1, 1, -1)
        X_train = torch.cat((X_train, static_attr.expand(X_train.shape[0], X_train.shape[1], -1)), dim=2)
        X_val = torch.cat((X_val, static_attr.expand(X_val.shape[0], X_val.shape[1], -1)), dim=2)
        X_test = torch.cat((X_test, static_attr.expand(X_test.shape[0], X_test.shape[1], -1)), dim=2)

    outputs = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    if return_scaler:
        outputs.append(pipeline.scaler)
    if return_feature_names:
        outputs.append(feature_names)
    return tuple(outputs)


class multi_reservoir_data:
    """Store data from multiple reservoirs"""

    def __init__(self, left_years_dict, res_list, storage=False, attributes=None):
        self.left_years_dict = left_years_dict
        self.res_list = res_list
        self.attributes = attributes
        self.storage = storage

        self.X_test_dict = {}
        self.y_test_dict = {}
        self.scaler_dict = {}
        self.feature_names = None

    def fetch_data(self):
        """Run data processing for each reservoir and save the final 20% test split"""
        for reservoir, left_year in tqdm(self.left_years_dict.items(), desc='Processing test data: '):
            result = data_processing(
                res_id=reservoir,
                transform_type='standardize',
                train_frac=0.6,
                val_frac=0.2,
                test_frac=0.2,
                left=f'{left_year}-01-01',
                right='2020-12-31',
                return_scaler=True,
                storage=self.storage,
                attributes=self.attributes,
                return_feature_names=True,
            )
            self.X_test_dict[reservoir] = result[2][0]
            self.y_test_dict[reservoir] = result[2][1]
            self.scaler_dict[reservoir] = result[3]
            if self.feature_names is None:
                self.feature_names = result[4]
        return

    def combine_test_data(self):
        """Concatenate all reservoir test tensors along chunk dimension"""
        X_test = torch.cat([self.X_test_dict[key] for key in self.res_list], dim=0)
        y_test = torch.cat([self.y_test_dict[key] for key in self.res_list], dim=0)
        return X_test, y_test


def infer_model_hyperparameters(state_dict):
    """
    Infer LSTMModel1_opt architecture from a saved checkpoint.
    Returns:
    dict of model keyword arguments
    """
    input_size = state_dict['lstm.weight_ih_l0'].shape[1]
    hidden_size1 = state_dict['lstm.weight_hh_l0'].shape[1]
    hidden_size2 = state_dict['linear1.weight'].shape[0]
    output_size = state_dict['linear2.weight'].shape[0]
    num_layers = len([key for key in state_dict if key.startswith('lstm.weight_ih_l')])
    return {
        'input_size': input_size,
        'hidden_size1': hidden_size1,
        'hidden_size2': hidden_size2,
        'output_size': output_size,
        'num_layers': num_layers,
        'dropout_prob': 0.3,
    }


def load_pooled_model(checkpoint_path, device):
    """
    Load pooled LSTM model from reviewer-comments checkpoint.
    """
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model_kwargs = infer_model_hyperparameters(state_dict)
    model = LSTMModel1_opt(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_feature_groups(feature_names):
    """
    Group one-hot encoded columns into interpretable categories.
    Returns:
    list of tuples [(group_name, feature_names_in_group), ...]
    """
    groups = []

    if 'inflow' in feature_names:
        groups.append(('inflow', ['inflow']))
    if 'doy' in feature_names:
        groups.append(('doy', ['doy']))
    if 'storage' in feature_names:
        groups.append(('storage', ['storage']))

    use_cols = [c for c in feature_names if c.startswith('USE_')]
    dor_cols = [c for c in feature_names if c.startswith('DOR_')]
    agency_cols = [c for c in feature_names if c.startswith('AGENCY_')]

    if use_cols:
        groups.append(('main use', use_cols))
    if dor_cols:
        groups.append(('DOR level', dor_cols))
    if agency_cols:
        groups.append(('agency', agency_cols))
    if 'CAP_MCM' in feature_names:
        groups.append(('capacity', ['CAP_MCM']))

    return groups


def permute_feature_group(X, group_indices, rng):
    """
    Permute a feature group across chunk samples while preserving the feature structure
    within each chunk.
    """
    perm = torch.as_tensor(rng.permutation(X.shape[0]), dtype=torch.long)
    X_perm = X.clone()
    X_perm[:, :, group_indices] = X[perm][:, :, group_indices]
    return X_perm


def compute_grouped_permutation_importance(model, X_test, y_test, feature_names, n_repeats=20, random_seed=0, device=torch.device("cpu")):
    """
    Compute grouped permutation feature importance on pooled test data.
    Importance is defined as the drop in test R2 after permuting a feature group.
    """
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    baseline_r2 = r2_score_tensor(model=model, X=X_test, y=y_test)
    feature_groups = build_feature_groups(feature_names=feature_names)
    rng = np.random.default_rng(random_seed)

    results = []
    for group_name, group_cols in tqdm(feature_groups, desc='Permutation importance: '):
        group_indices = [feature_names.index(col) for col in group_cols]
        permuted_r2 = []
        for _ in range(n_repeats):
            X_perm = permute_feature_group(X=X_test, group_indices=group_indices, rng=rng)
            permuted_r2.append(r2_score_tensor(model=model, X=X_perm, y=y_test))

        permuted_r2 = np.array(permuted_r2)
        importance = baseline_r2 - permuted_r2
        results.append(
            {
                'feature_group': group_name,
                'feature_columns': ', '.join(group_cols),
                'baseline_r2': baseline_r2,
                'permuted_r2_mean': permuted_r2.mean(),
                'permuted_r2_std': permuted_r2.std(ddof=1),
                'importance_mean': importance.mean(),
                'importance_std': importance.std(ddof=1),
            }
        )

    results_df = pd.DataFrame(results).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return results_df


def plot_feature_importance(results_df, figure_path):
    """
    Create a bar chart in the same simple Matplotlib style used elsewhere in reviewer comments.
    """
    fig, ax = plt.subplots()
    ax.bar(
        results_df['feature_group'],
        results_df['importance_mean'],
        yerr=results_df['importance_std'],
        color='lightgrey',
        edgecolor='black',
        linewidth=0.8,
        capsize=4,
    )
    ax.set_ylabel('Drop in Test $R^2$ (OOS Reservoirs)', size='x-large')
    ax.set_xlabel('Feature Group', size='x-large')
    ax.set_title('Grouped Permutation Importance for Pooled Model', size='x-large')
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close(fig)
    return


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cpu")
    print(f'Using device: {device}')

    # Match the reviewer-comments pooled training setup
    attribute_df = get_attributes(index_type=int)
    res_list = pd.read_csv(OOS_RESULTS_PATH, index_col=0).index.to_list()
    left_years_dict = get_left_years(res_list=res_list)

    test_data = multi_reservoir_data(left_years_dict=left_years_dict, res_list=res_list, attributes=attribute_df)
    test_data.fetch_data()
    X_test, y_test = test_data.combine_test_data()

    model = load_pooled_model(checkpoint_path=CHECKPOINT_PATH, device=device)
    importance_df = compute_grouped_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=test_data.feature_names,
        n_repeats=20,
        random_seed=0,
        device=device,
    )

    importance_df.to_csv(SUMMARY_PATH, index=False)
    plot_feature_importance(results_df=importance_df, figure_path=FIGURE_PATH)

    print(f'Saved feature importance summary to: {SUMMARY_PATH}')
    print(f'Saved feature importance figure to: {FIGURE_PATH}')
    return


if __name__ == '__main__':
    main()
