import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import optuna

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.utils import plot_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
HIDDEN_LAYERS = 5
NEURONS_PER_LAYER = [4096, 2048, 1024, 256, 256]
EPOCHS = 115
OPTIMIZER = 'sgd'
LEARNING_RATE = 0.01
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()

train_dataset, validation_dataset = load_data(DATASET_DIR, IMG_SIZE, BATCH_SIZE, preprocessing_method = 'rescaling')

model = build_model(IMG_SIZE, HIDDEN_LAYERS, NEURONS_PER_LAYER, None, None, OPTIMIZER, LEARNING_RATE)

print(model.summary())
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

history = model.fit(train_dataset, epochs=EPOCHS,validation_data=validation_dataset, verbose=0)

print('Saving the model')
model.save("model_mlp.h5")

train_img = []
train_lb = []

for img, lb in train_dataset:
    train_img.append(img.numpy())
    train_lb.append(np.argmax(lb.numpy(), axis=1) )
train_img = np.concatenate(train_img)
train_lb = np.concatenate(train_lb)

test_img = []
test_lb = []

for img, lb in validation_dataset:
    test_img.append(img.numpy())
    test_lb.append(np.argmax(lb.numpy(), axis=1) )
test_img = np.concatenate(test_img)
test_lb = np.concatenate(test_lb)

study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "task4", direction='maximize')
study.optimize(lambda trial: search_svm_optuna(trial, model, train_img, train_lb, test_img, test_lb), n_trials=100, n_jobs=1)

print('Done!')




