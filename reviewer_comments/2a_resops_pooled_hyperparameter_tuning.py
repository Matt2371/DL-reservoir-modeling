# Workaround: add directory of 'src' and 'ssjrb_wrapper' to the sys.path
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(file_dir, '..')) # One level up to the project root
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from math import floor
import os
import copy
from tqdm import tqdm
import geopandas as gpd

from src.data.data_processing import *
from src.data.data_fetching import *
from src.models.model_zoo import *
from src.models.predict_model import *
from src.models.train_model import *

## --------      GRID SEARCH FOR POOLED RESOPS MODEL, ADD STATIC ATTRIBUTES FROM GRAND    ------- ##

gdf = gpd.read_file("data/GRanD/GRanD_dams_v1_3.shp")
gdf = gdf.drop(columns="geometry").set_index("GRAND_ID")
print(gdf.columns)