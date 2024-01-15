import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *

import matplotlib
matplotlib.use('Agg')


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

train_dataset, validation_dataset = load_data(DATASET_DIR, IMG_SIZE, 16, preprocessing_method = 'rescaling')

test_neurons = [[4096, 4096, 2048],
                [4096, 4096, 1024],
                [4096, 4096, 256],
                [4096, 2048, 1024],
                [4096, 2048, 256]]

for neurons in test_neurons:
    model = build_model(IMG_SIZE, 3, neurons, None, None, 'sgd', 0.01)
    history = model.fit(train_dataset, epochs=50,validation_data=validation_dataset, verbose=0)
    print(history.history['val_accuracy'][-1])




