#### LINEAR PROBE FOR POOLED AND HORIZON-SPECIFIC OOS MODELS ####

# Workaround: add directory of 'src' and 'ssjrb_wrapper' to the sys.path
import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..'))  # One level up to the project root
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.data_processing import *
from src.data.data_fetching import *
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import *
from src.models.hyperparameter_tuning import *
from src.models.analyze_lstm_cell import *

from reviewer_comments.pooled_training_utils import (
    get_attributes,
    data_processing,
    multi_reservoir_data_oos,
    multi_reservoir_data,
)


OOS_LIST_PATH = 'report/results/reviewer_comments/hyperparameter_tuning_pooled/resops_oos_out_of_sample_test.csv'
POOLED_GRID_SEARCH_PATH = 'report/results/reviewer_comments/hyperparameter_tuning_pooled/grid_search_model1_pooled.csv'
POOLED_UNROLL_MODEL_PATH = 'report/results/reviewer_comments/hyperparameter_tuning_pooled/resops_simul_unroll_model.pt'
SAVE_DIR = 'report/results/reviewer_comments/pooled_linear_probe'
FINETUNE_YEARS = [5, 10, 15, 20, 25, 30]


class CustomLinearProbe(cell_linear_probe):
    def __init__(self):
        super().__init__()

    def fit_probe(self, cell_states, storage_state):
        # Override fit_probe to handle flattening and removing pad outside of method
        self.model.fit(cell_states, storage_state)
        self.isfit = True
        return


def get_best_unroll_model_kwargs(input_size):
    grid_search_df = pd.read_csv(POOLED_GRID_SEARCH_PATH, index_col=0)
    best_params = (
        grid_search_df
        .groupby(['num_layers', 'hidden1', 'hidden2', 'dropout'], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=['random_seed'], errors='ignore')
        .sort_values(by='val_error', axis=0, ascending=True)
        .iloc[0, :]
    )
    print(f"Best hyperparameters: {best_params}")

    return {
        'input_size': input_size,
        'hidden_size1': int(best_params.hidden1),
        'hidden_size2': int(best_params.hidden2),
        'output_size': 1,
        'num_layers': int(best_params.num_layers),
        'dropout_prob': float(best_params.dropout),
    }


def train_horizon_model(res_id, left_year, nyears, model_kwargs, attributes=None, pretrained=False, device=torch.device("cpu")):
    """
    Train an unrolled model on the first n years of a reservoir record.
    If pretrained=True, initialize from the pooled unrolled checkpoint before finetuning.
    Returns the trained model, horizon-window training inputs, and corresponding observed storage.
    """
    right_date = f'{left_year + nyears - 1}-12-31'
    train_data, val_data, _ = data_processing(
        res_id=res_id,
        transform_type='standardize',
        train_frac=0.75,
        val_frac=0.25,
        test_frac=0.0,
        left=f'{left_year}-01-01',
        right=right_date,
        attributes=attributes,
    )
    df_train, _, _ = split_df_data(
        id=res_id,
        left=f'{left_year}-01-01',
        right=right_date,
        data_splits=(0.75, 0.25, 0.0),
        fill_train_mean=True,
    )

    # train_data / val_data are tuples of (X, y) with shapes
    # (# chunks, chunk_size, # features) and (# chunks, chunk_size, 1)
    torch.manual_seed(0)
    model = LSTMModel1(**model_kwargs)
    if pretrained:
        model.load_state_dict(torch.load(POOLED_UNROLL_MODEL_PATH, weights_only=True, map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset_train = TensorDataset(*train_data)
    dataset_val = TensorDataset(*val_data)
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

    training_loop(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        patience=10,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        epochs=1000,
        device=device,
    )

    return model, train_data[0], df_train['storage'].values.flatten()  # X_train: (# chunks, chunk_size, # features); storage: (# timesteps,)


def fit_layerwise_probes(model, X_train, storage_train, num_layers, device):
    probes = [CustomLinearProbe() for _ in range(num_layers)]
    with torch.no_grad():
        model.eval()
        _, (_, cell_train) = model(X_train.to(device))
        # cell_train is a list of length num_layers; each entry is
        # (# chunks, chunk_size, hidden_size1)

    for layer_idx, probe in enumerate(probes):
        # Flatten to align chunked cell states with the 1D observed storage series.
        flattened_cells = flatten_cells(cell_train[layer_idx].to('cpu'))[:len(storage_train), :]
        probe.fit_probe(flattened_cells, storage_train)

    return probes


def fit_pooled_layerwise_probes(model, X_train_dict, storage_train_dict, reservoir_list, num_layers, device):
    probes = [CustomLinearProbe() for _ in range(num_layers)]
    pooled_cells_by_layer = [[] for _ in range(num_layers)]
    pooled_storage = []

    for res in reservoir_list:
        storage_train = storage_train_dict[res]
        pooled_storage.append(storage_train)

        with torch.no_grad():
            model.eval()
            _, (_, cell_train) = model(X_train_dict[res].to(device))
            # X_train_dict[res]: (# chunks, chunk_size, # features)
            # cell_train[layer]: (# chunks, chunk_size, hidden_size1)

        for layer_idx in range(num_layers):
            # Trim pad at the reservoir level before pooling across reservoirs.
            pooled_cells_by_layer[layer_idx].append(
                flatten_cells(cell_train[layer_idx].to('cpu'))[:len(storage_train), :]
            )

    pooled_storage = np.concatenate(pooled_storage, axis=0)
    for layer_idx, probe in enumerate(probes):
        probe.fit_probe(np.concatenate(pooled_cells_by_layer[layer_idx], axis=0), pooled_storage)

    return probes


def evaluate_layerwise_probes(model, probes, X_eval, storage_eval, device):
    with torch.no_grad():
        model.eval()
        _, (_, cell_eval) = model(X_eval.to(device))
        # X_eval is (# chunks, chunk_size, # features); storage_eval is (# timesteps,)

    return [
        probes[layer_idx].correlate_prediction(cell_eval[layer_idx].to('cpu'), storage_eval)
        for layer_idx in range(len(probes))
    ]


def format_horizon_probe_results(results_dict, year_list, num_layers, model_prefix):
    columns = {}
    for nyears in year_list:
        for layer_idx in range(num_layers):
            columns[f'{model_prefix}_{nyears}yr_layer_{layer_idx + 1}_corr'] = [
                results_dict[res][nyears][layer_idx] for res in results_dict
            ]
    return pd.DataFrame(columns, index=list(results_dict.keys()))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Set the device
    device = get_device()
    print(f"Using device: {device}")

    # -----   Data collection and processing    ----- #

    attribute_df = get_attributes(index_type=str)
    res_list = filter_res()
    oos_list = pd.read_csv(OOS_LIST_PATH, index_col=0).index.astype(str).to_list()
    left_years_dict = get_left_years(res_list=res_list)

    # For fitting pooled probe on in-sample reservoirs (0.75/0.25 train/val split, ignore test set)
    ios_data_combiner = multi_reservoir_data_oos(
        left_years_dict=left_years_dict,
        res_list=res_list,
        oos_list=oos_list,
        attributes=attribute_df,
    )
    ios_data_result = ios_data_combiner.fetch_and_combine()

    ios_list = [res for res in res_list if res not in oos_list]
    assert 0 < len(ios_list) < len(res_list)
    train_storage_dict = {}

    for res in tqdm(ios_list, desc='Collecting in-sample training data for pooled probe: '):
        df_train, _, _ = split_df_data(
            id=res,
            left=f'{left_years_dict[res]}-01-01',
            data_splits=(0.75, 0.25, 0.0),
            fill_train_mean=True,
        )
        train_storage_dict[res] = df_train['storage'].values.flatten()

    # For evaluating probes on the last 20% of OOS records
    oos_data_combiner = multi_reservoir_data(
        left_years_dict=left_years_dict,
        res_list=oos_list,
        attributes=attribute_df,
    )
    oos_data_combiner.fetch_data()

    test_storage_dict = {}
    for res in oos_list:
        _, _, df_test = split_df_data(
            id=res,
            left=f'{left_years_dict[res]}-01-01',
            data_splits=(0.6, 0.2, 0.2),
            fill_train_mean=True,
        )
        # Observed storage is kept as a flat 1D series so it lines up with flattened cell states.
        test_storage_dict[res] = df_test['storage'].values.flatten()

    # -----   Fit pooled unrolled model probe     ----- #

    model_kwargs = get_best_unroll_model_kwargs(input_size=ios_data_result[0][0].shape[2])
    num_layers = model_kwargs['num_layers']

    simul_unroll_model = LSTMModel1(**model_kwargs)
    simul_unroll_model.load_state_dict(torch.load(POOLED_UNROLL_MODEL_PATH, weights_only=True, map_location=device))
    simul_unroll_model.to(device)

    pooled_probes = fit_pooled_layerwise_probes(
        model=simul_unroll_model,
        X_train_dict=ios_data_combiner.X_train_dict,
        storage_train_dict=train_storage_dict,
        reservoir_list=ios_list,
        num_layers=num_layers,
        device=device,
    )

    pooled_probe_results = []
    for res in tqdm(oos_list, desc='Evaluating pooled probe on OOS test data: '):
        pooled_probe_results.append(
            evaluate_layerwise_probes(
                model=simul_unroll_model,
                probes=pooled_probes,
                X_eval=oos_data_combiner.X_test_dict[res],
                storage_eval=test_storage_dict[res],
                device=device,
            )
        )

    pooled_probe_corr_df = pd.DataFrame(
        {
            f'layer_{layer_idx + 1}_pooled_corr': [
                pooled_probe_results[res_idx][layer_idx] for res_idx in range(len(oos_list))
            ]
            for layer_idx in range(num_layers)
        },
        index=oos_list,
    )
    pooled_probe_corr_df.to_csv(os.path.join(SAVE_DIR, 'oos_pooled_linear_probe_correlation.csv'))

    # -----   Fit horizon-window probes for baseline and finetuned pooled models    ----- #

    baseline_results = {res: {} for res in oos_list}
    finetuned_results = {res: {} for res in oos_list}

    for nyears in FINETUNE_YEARS:
        for res in tqdm(oos_list, desc=f'Processing {nyears}yr horizon models: '):
            baseline_model, X_train, storage_train = train_horizon_model(
                res_id=res,
                left_year=left_years_dict[res],
                nyears=nyears,
                model_kwargs=model_kwargs,
                attributes=attribute_df,
                pretrained=False,
                device=device,
            )
            baseline_model.to(device)
            baseline_probes = fit_layerwise_probes(
                model=baseline_model,
                X_train=X_train,
                storage_train=storage_train,
                num_layers=num_layers,
                device=device,
            )
            baseline_results[res][nyears] = evaluate_layerwise_probes(
                model=baseline_model,
                probes=baseline_probes,
                X_eval=oos_data_combiner.X_test_dict[res],
                storage_eval=test_storage_dict[res],
                device=device,
            )

            finetuned_model, X_train, storage_train = train_horizon_model(
                res_id=res,
                left_year=left_years_dict[res],
                nyears=nyears,
                model_kwargs=model_kwargs,
                attributes=attribute_df,
                pretrained=True,
                device=device,
            )
            finetuned_model.to(device)
            finetuned_probes = fit_layerwise_probes(
                model=finetuned_model,
                X_train=X_train,
                storage_train=storage_train,
                num_layers=num_layers,
                device=device,
            )
            finetuned_results[res][nyears] = evaluate_layerwise_probes(
                model=finetuned_model,
                probes=finetuned_probes,
                X_eval=oos_data_combiner.X_test_dict[res],
                storage_eval=test_storage_dict[res],
                device=device,
            )

    format_horizon_probe_results(
        results_dict=baseline_results,
        year_list=FINETUNE_YEARS,
        num_layers=num_layers,
        model_prefix='baseline',
    ).to_csv(os.path.join(SAVE_DIR, 'oos_baseline_linear_probe_correlation.csv'))

    format_horizon_probe_results(
        results_dict=finetuned_results,
        year_list=FINETUNE_YEARS,
        num_layers=num_layers,
        model_prefix='finetuned_pooled',
    ).to_csv(os.path.join(SAVE_DIR, 'oos_finetuned_pooled_linear_probe_correlation.csv'))

    return


# run script
if __name__ == '__main__':
    main()
