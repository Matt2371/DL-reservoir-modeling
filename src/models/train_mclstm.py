import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

#### IMPLEMENTS TRAINING LOOP FOR MC-LSTM WITH EARLY STOPPING ####

def create_mask(x, pad_value=-1):
    """Create mask to hide padded outputs from the loss function"""

    # x is same shape as TARGETS from dataloader; (batchsize, chunk_size, 1)
    mask = (x != pad_value).to(torch.float)
    return mask


def train_one_epoch_mclstm(model, criterion, optimizer, dataloader_train):
    """
    Same as train_one_epoch() except for MC-LSTM, where mass input and
    auxillary input need to be separated.
    Return training loss (averaged for over each minibatch) for the epoch
    """

    # training loop
    total_loss = 0
    model.train() # set model to training mode
    for inputs_m, inputs_a, targets in dataloader_train:
        optimizer.zero_grad()

        # Forward pass
        outputs, c = model(inputs_m, inputs_a)
        # Sum mass outputs to get prediction
        outputs = outputs.sum(dim=-1, keepdim=True) 

        # Apply mask on targets/outputs
        mask = create_mask(targets)
        masked_outputs = outputs * mask
        masked_targets = targets * mask

        # Calculate loss with masked targets
        loss = criterion(masked_outputs, masked_targets)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader_train)
    return avg_loss


def val_one_epoch_mclstm(model, criterion, dataloader_val):
    """
    Same as val_one_epoch() except for MC-LSTM, where mass input and
    auxillary input need to be separated.
    Return validation loss (averaged for over each minibatch) for the epoch
    """

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_val_loss = 0
        for val_inputs_m, val_inputs_a, val_targets in dataloader_val:
            # Forward pass
            val_outputs, c = model(val_inputs_m, val_inputs_a)
            # Sum mass outputs to get prediction
            val_outputs = val_outputs.sum(dim=-1, keepdim=True) 

            # Apply mask on targets/outputs
            mask = create_mask(val_targets)
            val_masked_outputs = val_outputs * mask
            val_masked_targets = val_targets * mask

            # Evaluate validation loss
            val_loss = criterion(val_masked_outputs, val_masked_targets)
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(dataloader_val)
    return avg_val_loss

class EarlyStopper:
    """Implement EarlyStopper callback to prevent overfitting"""
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def training_loop_mclstm(model, criterion, optimizer, patience, dataloader_train, dataloader_val, epochs):
    """
    Run full training loop with early stopping for MC-LSTM.
    Params:
    model -- PyTorch model to train
    criterion -- function to evaluate loss
    optimizer -- PyTorch optimizer
    patience -- number of patience epochs for the early stopper
    dataloader_train -- PyTorch Dataloader for training data
    dataloader_val -- PyTorch Dataloader for validation data
    epochs -- maximum number of training epochs

    Returns:
    train_losses -- average training losses for each epoch
    val_losses -- average validation losses for each epoch
    """

    num_epochs = epochs
    train_losses = [] # keep track of training and val losses for Model 1
    val_losses = []
    early_stopper = EarlyStopper(patience=patience) # instantiate early stopper
    for epoch in tqdm(range(num_epochs), desc='Training epochs: '):
        model.train() # set model1 to training mode
        train_loss = train_one_epoch_mclstm(model, criterion, optimizer, dataloader_train)
        train_losses.append(train_loss)
        val_loss = val_one_epoch_mclstm(model, criterion, dataloader_val)
        val_losses.append(val_loss)

        if early_stopper.early_stop(val_loss):
            break
    return train_losses, val_losses


