import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Add, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Activation, LeakyReLU, Dropout
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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os,sys
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Reshape, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Add, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Activation, LeakyReLU
from keras.activations import elu, selu

def optimizers_config(optimizer, lr):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'nadam':
        return tf.keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer == 'adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=lr)

def start1(inputs, activation, regularizer, initializer):
    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def separable_block(x, filters, strides, activation, regularizer, initializer):
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', depthwise_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def activate_layer(x, activation):
    if activation == 'relu':
        return ReLU()(x)
    if activation == 'leakyrelu':
        return LeakyReLU()(x)
    if activation == 'elu':
        return elu(x)
    if activation == 'selu':
        return selu(x)

def optuna_model_s4_1(trial, input_shape, classes, activation, regularizer, initializer):

    inputs = Input(shape=input_shape)

    x = start1(inputs, activation, regularizer, initializer)

    for i in range(7):
        if i % 2 == 0:
            stride = (1, 1)
        else:
            stride = (2, 2)
        
        filters = trial.suggest_categorical(f'block_{i}_filters', [32, 64, 256])
        x = separable_block(x, filters, stride, activation, regularizer, initializer)
    
    x = GlobalAveragePooling2D()(x)
    
    neurons = trial.suggest_categorical(f'neurons', [0, 8, 16, 24])
    
    if neurons != 0:
        drop = trial.suggest_categorical(f'dropout', [0.0, 0.1, 0.2])
        if drop != 0.0:
            x = Dropout(0.1)(x)
        x = Dense(neurons, kernel_regularizer=regularizer,  kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = activate_layer(x, activation)

    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def objective(trial, input_shape, train_dataset, validation_dataset, classes, epochs, optimizer, lr, activation, regularizer, initializer):
    
    opt= optimizers_config(optimizer, lr)

    model = optuna_model_s4_1(trial, input_shape, classes, activation, regularizer, initializer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if model.count_params() > 70000: 
        raise optuna.exceptions.TrialPruned()
    
    checkpoint = ModelCheckpoint('best_model_s4_1.keras', save_best_only=True, monitor='val_accuracy', mode='max')

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=[checkpoint], 
        verbose = 0
    )
     
    trial.set_user_attr('num_parameters', model.count_params())
    trial.set_user_attr('ratio', max(history.history['val_accuracy']) / (model.count_params()/100000))

    return max(history.history['val_accuracy'])


BATCH_SIZE= 8
ACTIVATION= 'leakyrelu'
REGULARIZER = None
INITIALIZER = 'he_normal'
IMG_HEIGHT=224
IMG_WIDTH = 224
OPTIMIZER='nadam'
LEARNING_RATE=0.001
EPOCHS = 150
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'


train_dataset, validation_dataset = load_data_augmented(DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

try:
    optuna.delete_study(storage='sqlite:///optuna.db', study_name='s4_1')
except:
    pass
study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "s4_1", direction='maximize')
study.optimize(lambda trial: objective(trial, (IMG_HEIGHT, IMG_WIDTH, 3), train_dataset, validation_dataset, 8, EPOCHS, OPTIMIZER, LEARNING_RATE, ACTIVATION, REGULARIZER, INITIALIZER), n_trials=60, n_jobs=1)

print(f"Best hyperparameters: {study.best_params} accuracy: {study.best_value}")
