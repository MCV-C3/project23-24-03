#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import tensorflow as tf
from keras.layers import Rescaling, RandomFlip, Normalization
from tensorflow.keras.regularizers import l1, l2
import tensorflow.keras as keras
from keras.layers import Dense, Reshape, Input
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)
 
  total = 2688
  count = 0  
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))
  
    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))
  
      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        print(im.size)
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        patches = image.extract_patches_2d(np.array(im), (64, 64), max_patches=1)
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')

def load_data(dataset_dir, img_size, batch_size, preprocessing_method, shuffle = True):
    # Load and preprocess the training dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
      directory=dataset_dir+'/train/',
      labels='inferred',
      label_mode='categorical',
      batch_size=batch_size,
      class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
      image_size=(img_size, img_size),
      shuffle=shuffle,
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
      shuffle=shuffle,
      seed=123,
      validation_split=None,
      subset=None
    )

    # Data augmentation and preprocessing
    preprocessing_train = keras.Sequential([
      RandomFlip("horizontal")
    ])

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

def build_model_optuna(trial, img_size):
    dense_layers = trial.suggest_categorical("dense_layers", [1, 2, 3, 4, 5])
    neurons_per_layer = [trial.suggest_categorical(f'n_neurons_layer_{i}', [32, 64, 128, 256, 512, 1024, 2048]) for i in range(dense_layers)]
    
    regularizer = trial.suggest_categorical("regularizer", [None, 'l2'])
    lr_r = trial.suggest_categorical("lr_r",[0.1, 0.01, 0.001])

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])

    #Build the Multi Layer Perceptron model
    model = Sequential()
    input = Input(shape=(img_size, img_size, 3,),name='input')
    model.add(input) # Input tensor
    model.add(Reshape((img_size*img_size*3,),name='reshape'))

    for i in range(dense_layers):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))

    model.add(Dense(units=8, activation='softmax',name='output'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    return model

def build_model_from_best_params(params, img_size, train_dataset, validation_dataset):
    dense_layers = params["dense_layers"]
    neurons_per_layer = [params[f'n_neurons_layer_{i}'] for i in range(dense_layers)]
    
    regularizer = params["regularizer"]
    lr_r = params["lr_r"]

    optimizer = params["optimizer"]
    lr_o = params["lr_o"]

    #Build the Multi Layer Perceptron model
    model = Sequential()
    input = Input(shape=(img_size, img_size, 3,),name='input')
    model.add(input) # Input tensor
    model.add(Reshape((img_size*img_size*3,),name='reshape'))

    for i in range(dense_layers):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))

    model.add(Dense(units=8, activation='softmax',name='output'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    model.fit(train_dataset, epochs=50, validation_data=validation_dataset, verbose=0)
    return model


def search_optuna(trial, img_size, train_dataset, validation_dataset):
    model = build_model_optuna(trial, img_size)
    history = model.fit(train_dataset, epochs=50, validation_data=validation_dataset, verbose=0)
    return history.history['val_accuracy'][-1]

def build_model(img_size, dense_layers, neurons_per_layer, regularizer, lr_r, optimizer, lr_o):

    #Build the Multi Layer Perceptron model
    model = Sequential()
    input = Input(shape=(img_size, img_size, 3,),name='input')
    model.add(input) # Input tensor
    model.add(Reshape((img_size*img_size*3,),name='reshape'))

    for i in range(dense_layers):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))

    model.add(Dense(units=8, activation='softmax',name='output'))

    model.compile(loss='categorical_crossentropy',optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])

    return model

def build_model_optuna_task2(trial, img_size):
    dense_layers = trial.suggest_categorical("dense_layers", [3, 4, 5])
    neurons_per_layer = [trial.suggest_categorical(f'n_neurons_layer_{i}', [64, 128, 256]) for i in range(dense_layers)]
    
    regularizer = trial.suggest_categorical("regularizer", [None, 'l2'])
    lr_r = trial.suggest_categorical("lr_r",[0.1, 0.01, 0.001])

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])

    #Build the Multi Layer Perceptron model
    model = Sequential()
    input = Input(shape=(img_size, img_size, 3,),name='input')
    model.add(input) # Input tensor
    model.add(Reshape((img_size*img_size*3,),name='reshape'))

    for i in range(dense_layers):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))

    model.add(Dense(units=8, activation='softmax',name='output'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    return model

def search_optuna_task2(trial, img_size, train_dataset, validation_dataset):
    model = build_model_optuna_task2(trial, img_size)
    history = model.fit(train_dataset, epochs=25, validation_data=validation_dataset, verbose=0)
    return history.history['val_accuracy'][-1]

def build_model_optuna_task2_1(trial, img_size):
    dense_layers = trial.suggest_categorical("dense_layers", [3, 4, 5])
    neurons_per_layer = [512, 512, 512, 512, 512]
    
    regularizer = trial.suggest_categorical("regularizer", [None, 'l2'])
    lr_r = trial.suggest_categorical("lr_r",[0.1, 0.01, 0.001])

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])

    #Build the Multi Layer Perceptron model
    model = Sequential()
    input = Input(shape=(img_size, img_size, 3,),name='input')
    model.add(input) # Input tensor
    model.add(Reshape((img_size*img_size*3,),name='reshape'))

    for i in range(dense_layers):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))

    model.add(Dense(units=8, activation='softmax',name='output'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    return model

def search_optuna_task2_1(trial, img_size, train_dataset, validation_dataset):
    model = build_model_optuna_task2_1(trial, img_size)
    history = model.fit(train_dataset, epochs=25, validation_data=validation_dataset, verbose=0)
    return history.history['val_accuracy'][-1]