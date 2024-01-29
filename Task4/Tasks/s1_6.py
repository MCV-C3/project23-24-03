
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

def optuna_model_s1_6(trial, input_shape, classes, activation, regularizer, initializer):

    inputs = Input(shape=input_shape)
    x = inputs

    ## calculate first residual
    num_conv = trial.suggest_int(f'n_conv_block_0', 1, 3)
    filters = trial.suggest_categorical(f'filters_block_0', [16, 32, 64, 128, 256])
    for conv in range(num_conv):
        x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    residual = x
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)

    n_blocks = trial.suggest_int(f'n_blocks', 2, 5)

    for i in range(1, n_blocks):
        num_conv = trial.suggest_int(f'n_conv_block_{i}', 1, 3)
        filters = trial.suggest_categorical(f'filters_block_{i}', [16, 32, 64, 128, 256])
        for conv in range(num_conv):
            x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
        residual = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(residual)
        x = Add()([residual, x])
        residual = x
        x = BatchNormalization()(x)
        x = activate_layer(x, activation)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model
 
def objective(trial, train_dataset, validation_dataset, epochs, input_shape, classes, activation, regularizer, initializer):
    model = optuna_model_s1_6(trial, input_shape, classes, activation, regularizer, initializer)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), metrics=['accuracy'])
    # checkpoint = ModelCheckpoint('best_model_s1_6.keras', save_best_only=True, monitor='val_accuracy', mode='max')
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
activation= 'relu'

train_dataset, validation_dataset = load_data_augmented(DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

try:
    optuna.delete_study(storage='sqlite:///optuna.db', study_name='s1_6')
except:
    pass
study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "s1_6", direction='maximize')
study.optimize(lambda trial: objective(trial, train_dataset, validation_dataset, NUMBER_OF_EPOCHS, (IMG_HEIGHT, IMG_WIDTH, 3), 8, activation, None, None), n_trials=50, n_jobs=1)

print(f"Best hyperparameters: {study.best_params} accuracy: {study.best_value}")
