import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#user defined variables
IMG_SIZE    = 32
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()

batch_sizes = [8, 16, 32, 64, 128]
epochs = [20, 40, 50, 60, 80, 100]
learning_rates = [0.1, 0.01, 0.001]

for lr in learning_rates:
    batch_accuracy = []
    for batch in batch_sizes:
        train_dataset, validation_dataset = load_data(DATASET_DIR, IMG_SIZE, batch, preprocessing_method = 'rescaling')
        epoch_accuracy = []
        for epoch in epochs:
            model = build_model(IMG_SIZE, 5, [4096, 2048, 1024, 256, 256], None, None, 'sgd', lr)
            history = model.fit(train_dataset, epochs=epoch,validation_data=validation_dataset, verbose=0)
            epoch_accuracy.append(history.history['val_accuracy'][-1])
        batch_accuracy.append(epoch_accuracy)

    print(f'learning rate {lr}')
    for i, v in enumerate(batch_accuracy):
        print(f'Batch size {batch_sizes[i]}: {v}')
        plt.plot(epochs, v, label=f'Batch {batch_sizes[i]}')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy with learning rate {lr}')
    plt.legend(loc='upper left')
    plt.xticks(epochs)
    plt.savefig(f'{lr}_batch_epochs.jpg')
    plt.show()
    plt.close()





