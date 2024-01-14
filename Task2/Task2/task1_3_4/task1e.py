import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#user defined variables
BATCH_SIZE  = 16
LEARNING_RATE = 0.01
HIDDEN_LAYERS = 5
NEURONS_PER_LAYER = [4096, 2048, 1024, 256, 256]
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()

epochs = [100, 115, 130, 145, 160]
image_sizes = [8, 16, 32, 64, 128]

img_accuracy = []
for img in image_sizes:
    train_dataset, validation_dataset = load_data(DATASET_DIR, img, BATCH_SIZE, preprocessing_method = 'rescaling')
    epoch_accuracy = []
    for epoch in epochs:
        model = build_model(img, HIDDEN_LAYERS, NEURONS_PER_LAYER, None, None, 'sgd', LEARNING_RATE)
        history = model.fit(train_dataset, epochs=epoch,validation_data=validation_dataset, verbose=0)
        epoch_accuracy.append(history.history['val_accuracy'][-1])
    img_accuracy.append(epoch_accuracy)


for i, v in enumerate(img_accuracy):
    print(f'Image size {image_sizes[i]}: {v}')
    plt.plot(epochs, v, label=f'Size {image_sizes[i]}')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Accuracy with different image size')
plt.legend(loc='upper left')
plt.xticks(epochs)
plt.savefig(f'img_size_epochs.jpg')
plt.show()
plt.close()




