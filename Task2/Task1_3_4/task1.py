import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import optuna

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()

print('Setting up data ...\n')

train_dataset, validation_dataset = load_data(DATASET_DIR, IMG_SIZE, BATCH_SIZE, preprocessing_method = 'rescaling')

print('Building MLP model...\n')

#Fixed parameters
#EPOCHS = 50
#IMG_SIZE = 32
#BATCH_SIZE = 16


study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "task1b", direction='maximize')
study.optimize(lambda trial: search_model_optuna(trial, 'architecture', IMG_SIZE, train_dataset, validation_dataset), n_trials=100, n_jobs=1)

print('Done!')




