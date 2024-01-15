#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import tensorflow as tf
from keras.layers.experimental.preprocessing import Rescaling, RandomFlip, Normalization
from tensorflow.keras.regularizers import l1, l2
import keras
from keras.layers import Dense, Reshape, Input
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from collections import Counter

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def voting(x):
    x_max = x.argmax(axis=1)
    return Counter(x_max).most_common(1)[0][0]

def softmax_mean(x):
    return np.argmax( softmax(np.mean(x, axis=0)))

def generate_image_patches_db(in_directory, out_directory, patch_size=64, max_patches=1):
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
        patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size), max_patches=max_patches)
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')

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

def load_data_patch(patches_dir, patch_size, batch_size, preprocessing_method):
    # Load and preprocess the training dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
    directory=patches_dir+'/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    image_size=(patch_size, patch_size)
    )

    # Load and preprocess the validation dataset
    validation_dataset = keras.preprocessing.image_dataset_from_directory(
    directory=patches_dir+'/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    image_size=(patch_size, patch_size)
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

def search_svm_optuna(trial, model, train_img, train_lb, test_img, test_lb):
    hidden_layer = trial.suggest_categorical("hidden_layer", [-2, -3, -4, -5])
    standardize = trial.suggest_categorical('standardize', [True, False])
    C = trial.suggest_float('C', 1.0, 4.0)

    model_layer = keras.Model(inputs=model.input, outputs=model.layers[hidden_layer].output)

    train_descriptor = model_layer.predict(train_img)
    test_descriptor = model_layer.predict(test_img)

    if standardize:
        scaler = StandardScaler()
        train_descriptor = scaler.fit_transform(train_descriptor)
        test_descriptor = scaler.transform(test_descriptor)

    svm = SVC(C = C, kernel = 'rbf')
    svm = svm.fit(train_descriptor, train_lb)

    return svm.score(test_descriptor, test_lb)

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

def build_model_optuna_architecture(trial, img_size):
    dense_layers = trial.suggest_categorical("dense_layers", [2, 3, 4, 5])
    neurons_per_layer = [trial.suggest_categorical(f'n_neurons_layer_{i}', [256, 512, 1024, 2048, 4096]) for i in range(dense_layers)]
    
    #Build the Multi Layer Perceptron model
    model = Sequential()
    input = Input(shape=(img_size, img_size, 3,),name='input')
    model.add(input) # Input tensor
    model.add(Reshape((img_size*img_size*3,),name='reshape'))

    for i in range(dense_layers):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=None, name=f'dense_layer_{i}'))

    model.add(Dense(units=8, activation='softmax',name='output'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config('sgd', 0.01), metrics=['accuracy'])
    return model

def search_model_optuna(trial, option, img_size, train_dataset, validation_dataset):

    if option == 'general': 
        model = build_model_optuna(trial, img_size)
    elif option == 'architecture':
        model = build_model_optuna_architecture(trial, img_size)
        
    history = model.fit(train_dataset, epochs=50, validation_data=validation_dataset, verbose=0)
    return history.history['val_accuracy'][-1]

def build_model(img_size, hidden_layer, neurons_per_layer, regularizer, lr_r, optimizer, lr_o):

    #Build the Multi Layer Perceptron model
    model = Sequential()
    input = Input(shape=(img_size, img_size, 3,),name='input')
    model.add(input) # Input tensor
    model.add(Reshape((img_size*img_size*3,),name='reshape'))

    for i in range(hidden_layer):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))

    model.add(Dense(units=8, activation='softmax',name='output'))

    model.compile(loss='categorical_crossentropy',optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])

    return model


def build_model_patch(patch_size, hidden_layer, neurons_per_layer, regularizer, lr_r, optimizer, lr_o, phase = 'train'):

    #Build the Multi Layer Perceptron model
    model = Sequential()
    input = Input(shape=(patch_size, patch_size, 3,),name='input')
    model.add(input) # Input tensor
    model.add(Reshape((patch_size*patch_size*3,),name='reshape'))

    for i in range(hidden_layer):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))

    if phase=='test':
        model.add(Dense(units=8, activation='linear',name='output'))# In test phase we softmax the average output over the image patches
    else:
        model.add(Dense(units=8, activation='softmax',name='output'))

    model.compile(loss='categorical_crossentropy',optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])

    return model