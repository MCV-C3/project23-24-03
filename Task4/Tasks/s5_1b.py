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
matplotlib.use('Agg')
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

def best_model(input_shape, classes, activation, regularizer, initializer):

    inputs = Input(shape=input_shape)

    x = start5(inputs, activation, regularizer, initializer)

    x = separable_block(x, 32, (1, 1), activation, regularizer, initializer)
    x = separable_block(x, 64, (2, 2), activation, regularizer, initializer)
    x = separable_block(x, 128, (1, 1), activation, regularizer, initializer)
    x = separable_block(x, 32, (2, 2), activation, regularizer, initializer)
    x = separable_block(x, 64, (1, 1), activation, regularizer, initializer)
    x = separable_block(x, 256, (2, 2), activation, regularizer, initializer)
    x = separable_block(x, 64, (1, 1), activation, regularizer, initializer)

    x = GlobalAveragePooling2D()(x)

    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

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
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
CLASSES = 8
FILE = '5_1b5'

train_dataset, validation_dataset = load_data_augmented(DATASET_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

opt= optimizers_config(OPTIMIZER, LEARNING_RATE)

model = best_model(INPUT_SHAPE, CLASSES, ACTIVATION, REGULARIZER, INITIALIZER)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model_'+FILE+'.keras', save_best_only=True, monitor='val_accuracy', mode='max')

print(model.count_params())
if model.count_params() < 70000:

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=[checkpoint], 
        verbose = 2
    )
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(FILE+'_accuracy.jpg')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(FILE+'_loss.jpg')
    plt.close()
    print(max(history.history['val_accuracy']))
    print(f'ratio {max(history.history["val_accuracy"]) / (model.count_params()/100000)}')
    print(FILE)