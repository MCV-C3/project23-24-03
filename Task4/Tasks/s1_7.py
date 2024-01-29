
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Add, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Activation, LeakyReLU
from keras.models import Model
from keras.activations import elu
from keras.layers.experimental.preprocessing import Rescaling, RandomFlip, Normalization
from keras.utils import plot_model
from tensorflow.keras.regularizers import l1, l2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import optuna 
from tensorflow.keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical


def optimizers_config(optimizer, lr):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'nadam':
        return tf.keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer == 'adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=lr)
 
def optuna_model_s1_7(trial, input_shape, classes, activation, regularizer, initializer):
    init_block = trial.suggest_categorical('init_block', ['s1', 's2', 's3', 's4', 's5'])
    n_blocks = trial.suggest_int(f'n_blocks', 2, 5)

    inputs = Input(shape=input_shape)

    if init_block == 's1':
        x = start1(inputs, activation, regularizer, initializer)
    if init_block == 's2':
        x = start2(inputs, activation, regularizer, initializer)
    if init_block == 's3':
        x = start3(inputs, activation, regularizer, initializer)
    if init_block == 's4':
        x = start4(inputs, activation, regularizer, initializer)
    if init_block == 's5':
        x = start5(inputs, activation, regularizer, initializer)

    for i in range(n_blocks):
        if i % 2 == 0:
            stride = (1, 1)
        else:
            stride = (2, 2)

        filters = trial.suggest_categorical(f'block_{i}_filters', [16, 32, 48, 64, 96, 128, 256])
        x = separable_block(x, filters, stride, activation, regularizer, initializer)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def objective(trial, train_dataset, validation_dataset, epochs, input_shape, classes, activation, regularizer, initializer):
    model = optuna_model_s1_7(trial, input_shape, classes, activation, regularizer, initializer)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), metrics=['accuracy'])
    # checkpoint = ModelCheckpoint('best_model_s1_7.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    history = model.fit(
        train_dataset[0], train_dataset[1],
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=[], 
        verbose = 0
    )
    
    trial.set_user_attr('num_parameters', model.count_params())
    trial.set_user_attr('ratio', max(history.history['val_accuracy']) / (model.count_params()/100000))

    return max(history.history['val_accuracy'])

BATCH_SIZE=16
NUMBER_OF_EPOCHS=50
IMG_HEIGHT=32
IMG_WIDTH = 32
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
activation= 'relu'

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0 
train_labels = to_categorical(train_labels, 100)
test_labels = to_categorical(test_labels, 100)
try:
    optuna.delete_study(storage='sqlite:///optuna.db', study_name='s1_7')
except:
    pass
study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "s1_7", direction='maximize')
study.optimize(lambda trial: objective(trial, (train_images, train_labels), (test_images, test_labels), NUMBER_OF_EPOCHS, (IMG_HEIGHT, IMG_WIDTH, 3), 100, activation, None, None), n_trials=50, n_jobs=1)

print(f"Best hyperparameters: {study.best_params} accuracy: {study.best_value}")