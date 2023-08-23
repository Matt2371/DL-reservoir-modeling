import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#### FUNCTIONS TO MAKE PREDICTIONS WITH TRAINED MODELS ####

def predict(model, x):
    """
    Make predictions from a trained model.
    Params:
    model -- trained PyTorch model
    x -- input features
    Returns:
    output -- model output on x
    """

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output, _ = model(x)
    
    return output


