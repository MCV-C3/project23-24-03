import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
from keras.layers import Dense, GlobalAveragePooling2D

from keras.models import Model

def load_data(dataset_dir, img_size, batch_size, preprocessing_method):
    # Load and preprocess the training dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=dataset_dir+'/train/',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(img_size, img_size),
        shuffle=True,
        validation_split=None,
        subset=None
    )

    # Load and preprocess the validation dataset
    validation_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=dataset_dir+'/test/',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(img_size, img_size),
        shuffle=True,
        seed=123,
        validation_split=None,
        subset=None
    )

    # train_dataset = train_dataset.take(int(len(train_dataset) * 0.1))
    # validation_dataset = validation_dataset.take(int(len(validation_dataset) * 0.1))

    # Data augmentation and preprocessing
    preprocessing_train = keras.Sequential(
        RandomFlip("horizontal")
    )

    preprocessing_validation = keras.Sequential()

    if preprocessing_method == 'rescaling':
        preprocessing_train.add(Rescaling(1./255))
        preprocessing_validation.add(Rescaling(1./255))
    elif preprocessing_method == 'normalization':
        preprocessing_train.add(Normalization()) #standardization z-score
        preprocessing_validation.add(Normalization())

    train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
    validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset


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

def build_model_optuna_dropout_test(trial):
    dense_layers = trial.suggest_categorical("dense_layers", [1, 2, 3, 4, 5])
    neurons_per_layer = [trial.suggest_categorical(f'n_neurons_layer_{i}', [32, 64, 128, 256, 512, 1024, 2048]) for i in range(dense_layers)]
    
    regularizer = trial.suggest_categorical("regularizer", [None, 'l2'])
    lr_r = trial.suggest_categorical("lr_r",[0.1, 0.01, 0.001])

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])
    
    p = trial.suggest_float('dropout', 0.0, 0.9, step=0.1)

    #Build the Multi Layer Perceptron model
    base_model = InceptionV3(weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False)
    base_model.trainable = False
    
    base_model.layers[196].output
    x = GlobalAveragePooling2D()(x)

    for i in range(dense_layers):
        x = Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}')(x)
        x = Dropout(p)(x)

    outputs = Dense(units=8, activation='softmax',name='output')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    return model

def search_optuna(trial, train_dataset, validation_dataset):
    model = build_model_optuna_dropout_test(trial)
    history = model.fit(train_dataset, epochs=20, validation_data=validation_dataset, verbose=0)
    return history.history['val_accuracy'][-1]


if __name__ == '__main__':
    DATASET_DIR = '../MIT_small_train_1'

    IMG_HEIGHT=299
    IMG_WIDTH = 299
    BATCH_SIZE=16
    NUMBER_OF_EPOCHS=20
    
    train_dataset, validation_dataset = load_data(DATASET_DIR, IMG_HEIGHT, BATCH_SIZE, preprocessing_method = 'rescaling')
    study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "task4_final", direction='maximize')
    study.optimize(lambda trial: search_optuna(trial, train_dataset, validation_dataset), n_trials=30, n_jobs=1)