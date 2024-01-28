import seaborn as sns
import matplotlib.pyplot as plt
from tf_explain.core.grad_cam import GradCAM

import optuna
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

import tensorflow.keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, Flatten
from keras.layers import Rescaling, RandomFlip, Normalization
from keras.utils import plot_model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Dropout, BatchNormalization

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D

from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def regularizers_config(regularizer, lr):
    if regularizer == 'l1':
        return l1(lr)
    elif regularizer == 'l2':
        return l2(lr)
    return None

def optimizers_config(optimizer, lr):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr)

def get_best_model():
    new_model = Sequential()
    
    new_model.add(Input((IMG_HEIGHT,IMG_WIDTH,3)))
    
    model = tf.keras.models.load_model('best_models/model_CIFAR_1')
    
    for i in model.layers[:-1]:
        i._name = i.name + 'model1'
        i.trainable = False
        new_model.add(i)
    
    return new_model

def create_optuna_model(trial):
    backbone = get_best_model()
    
    dense_layers = trial.suggest_categorical("dense_layers", [1, 2, 3, 4, 5])
    neurons_per_layer = [trial.suggest_categorical(f'n_neurons_layer_{i}', [32, 64, 128, 256]) for i in range(dense_layers)]
    
    regularizer = trial.suggest_categorical("regularizer", [None, 'l2'])
    lr_r = trial.suggest_categorical("lr_r",[0.1, 0.01, 0.001])

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])
    
    p = trial.suggest_float('Dropout', 0.0, 0.6, step=0.1)

    #Build the Multi Layer Perceptron model

    for i in range(dense_layers):
        backbone.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))
        backbone.add(Dropout(p))

    backbone.add(Dense(units=8, activation='softmax',name='output'))
    
    backbone.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    print(backbone.summary())
    
    return backbone


def objective(trial):
    model = create_optuna_model(trial)
        
    history = model.fit(
        train_dataset,
        epochs=NUMBER_OF_EPOCHS,
        validation_data=validation_dataset    
    )
    
    return history.history['val_accuracy'][-1]


if __name__ == '__main__':
    DATASET_DIR = '../../MIT_small_train_1'
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    BATCH_SIZE = 16
    NUMBER_OF_EPOCHS = 50

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_dataset = train_datagen.flow_from_directory(
        directory=DATASET_DIR+'/train/',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_dataset = validation_datagen.flow_from_directory(
        directory=DATASET_DIR+'/test/',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    name_study = 'bestFinalArch2'
    try:
        optuna.delete_study(storage='sqlite:///optuna.db', study_name=name_study)
    except:
        pass
    
    study = optuna.create_study(storage='sqlite:///optuna.db', study_name=name_study, direction='maximize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_accuracy = study.best_value

    print(f"Best hyperparameters: {best_params}")
    print(f"Best accuracy: {best_accuracy}")
