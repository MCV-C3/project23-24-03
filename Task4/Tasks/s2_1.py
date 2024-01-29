
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


def optimizers_config(optimizer, lr):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'nadam':
        return tf.keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer == 'adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=lr)

def filter_by_block(trial, i):
    if i == 0:
        return trial.suggest_categorical(f'block_{i}_filters', [16, 32, 256])
    if i == 1:
        return trial.suggest_categorical(f'block_{i}_filters', [16, 32, 64, 256])
    if i == 2:
        return trial.suggest_categorical(f'block_{i}_filters',  [16, 32, 64, 256])
    if i == 3:
        return trial.suggest_categorical(f'block_{i}_filters',  [16, 32, 64, 256])
    if i == 4:
        return trial.suggest_categorical(f'block_{i}_filters',  [32, 64, 96, 256])  
    if i == 5:
        return trial.suggest_categorical(f'block_{i}_filters', [32, 64, 96, 256])  
    if i == 6:
        return trial.suggest_categorical(f'block_{i}_filters', [32, 64, 96, 256])  
    if i == 7:
        return trial.suggest_categorical(f'block_{i}_filters', [16, 32, 64, 128, 256])  
    if i == 8:
        return trial.suggest_categorical(f'block_{i}_filters', [16, 32, 96, 128, 256])      
    if i == 9:
        return trial.suggest_categorical(f'block_{i}_filters', [16, 32, 128, 256, 512])  
    if i == 10:
        return trial.suggest_categorical(f'block_{i}_filters', [16, 32, 128, 256, 512]) 

def optuna_model_s2_1(trial, input_shape, classes, activation, regularizer, initializer):

    n_blocks = trial.suggest_int(f'n_blocks', 5, 10)

    inputs = Input(shape=input_shape)

    x = start1(inputs, activation, regularizer, initializer)

    for i in range(n_blocks):
        if i % 2 == 0:
            stride = (1, 1)
        else:
            stride = (2, 2)

        filters = filter_by_block(trial, i) 
        x = separable_block(x, filters, stride, activation, regularizer, initializer)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def objective(trial, train_dataset, validation_dataset, epochs, input_shape, classes):
    activation = trial.suggest_categorical(f'activation', ["elu", "relu", "leakyrelu"])
    regularizer = trial.suggest_categorical(f'regularizer', [None, "l2"])
    initializer = trial.suggest_categorical(f'initializer', ["he_normal", "he_uniform", "glorot_uniform", "glorot_normal"])

 
    model = optuna_model_s2_1(trial, input_shape, classes, activation, regularizer, initializer)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), metrics=['accuracy'])

    if model.count_params() > 100000: 
        raise optuna.exceptions.TrialPruned()

    # checkpoint = ModelCheckpoint('best_model_s2_1.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    history = model.fit(
        train_dataset,
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
IMG_HEIGHT=224
IMG_WIDTH = 224
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'

train_dataset, validation_dataset = load_data_augmented(DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

try:
    optuna.delete_study(storage='sqlite:///optuna.db', study_name='s2_1')
except:
    pass
study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "s2_1", direction='maximize')
study.optimize(lambda trial: objective(trial, train_dataset, validation_dataset, NUMBER_OF_EPOCHS, (IMG_HEIGHT, IMG_WIDTH, 3), 8), n_trials=150, n_jobs=1)

print(f"Best hyperparameters: {study.best_params} accuracy: {study.best_value}")
