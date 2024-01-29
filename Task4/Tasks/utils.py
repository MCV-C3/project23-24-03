#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Reshape, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Add, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Activation, LeakyReLU
from keras.activations import elu, selu

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

def load_data(dataset_dir, img_size, batch_size, preprocessing_method='rescaling'):
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

def load_data_augmented(dataset_dir, img_height, img_width, batch_size):
    train_data_generator = ImageDataGenerator(featurewise_center=False,
                                              samplewise_center=False,
                                              featurewise_std_normalization=False,
                                              samplewise_std_normalization=False,
                                              rotation_range=0,
                                              width_shift_range=0.2,
                                              height_shift_range=0.,
                                              shear_range=0.2,
                                              brightness_range=[0.7, 1.3],
                                              zoom_range=0.,
                                              fill_mode='nearest',
                                              horizontal_flip=True,
                                              vertical_flip=False,
                                              rescale= 1./255)

    validation_data_generator = ImageDataGenerator(rescale= 1./255)

    train_dataset = train_data_generator.flow_from_directory(
        directory=dataset_dir+'/train/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=dataset_dir+'/test/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    return train_dataset, validation_dataset

######################
def start1(inputs, activation, regularizer, initializer):
    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def start2(inputs, activation, regularizer, initializer):
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def start3(inputs, activation, regularizer, initializer):
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def start4(inputs, activation, regularizer, initializer):
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def start5(inputs, activation, regularizer, initializer):
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def start1_res(inputs, activation, regularizer, initializer):
    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x, x

def start2_res(inputs, activation, regularizer, initializer):
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x, x

def start3_res(inputs, activation, regularizer, initializer):
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x, x

def start4_res(inputs, activation, regularizer, initializer):
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x, x

def start5_res(inputs, activation, regularizer, initializer):
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x, x

######################
def separable_block(x, filters, strides, activation, regularizer, initializer):
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', depthwise_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)
    return x

def separable_res_block(x, residual, filters, strides, activation, regularizer, initializer):
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = activate_layer(x, activation)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    residual = Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(residual)
    x = Add()([residual, x])
    x = activate_layer(x, activation)
    return x, x

######################
def activate_layer(x, activation):
    if activation == 'relu':
        return ReLU()(x)
    if activation == 'leakyrelu':
        return LeakyReLU()(x)
    if activation == 'elu':
        return elu(x)
    if activation == 'selu':
        return selu(x)

######################
def seprable_inception_res_block1(x, residual, activation, regularizer, initializer):
    #### branch 1
    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)

    #### branch 2
    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(24, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(48, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)

    #### branch 3
    branch3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    branch3 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch3)

    #### branch pool
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)

    branch_pool = Conv2D(16, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch_pool)
    
    ####
    mixed = concatenate([branch1, branch2, branch3, branch_pool], axis=-1)
    residual = Conv2D((32+48+32+16), (1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(residual)
    mixed = Add()([residual, mixed])
    residual_block = mixed
    mixed = BatchNormalization()(mixed)
    mixed = activate_layer(mixed, activation)

    return mixed, residual_block

def seprable_inception_res_block2(x, residual, activation, regularizer, initializer):
    #### branch 1
    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)

    #### branch 2
    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(48, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(96, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)

    #### branch 3
    branch3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    branch3 = Conv2D(96, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch3)

    #### branch pool
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)

    branch_pool = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch_pool)
    
    ####
    mixed = concatenate([branch1, branch2, branch3, branch_pool], axis=-1)
    residual = Conv2D((64+96+96+32), (1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(residual)

    mixed = Add()([residual, mixed])
    residual_block = mixed

    mixed = BatchNormalization()(mixed)
    mixed = activate_layer(mixed, activation)

    return mixed, residual_block

def seprable_inception_res_block3(x, residual, activation, regularizer, initializer):
    #### branch 1
    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)

    #### branch 2
    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(96, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)

    #### branch 3
    branch3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    branch3 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch3)

    #### branch pool
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)

    branch_pool = Conv2D(96, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch_pool)
    
    ####
    mixed = concatenate([branch1, branch2, branch3, branch_pool], axis=-1)
    residual = Conv2D((128+96+128+96), (1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(residual)

    mixed = Add()([residual, mixed])
    residual_block = mixed

    mixed = BatchNormalization()(mixed)
    mixed = activate_layer(mixed, activation)

    return mixed, residual_block

def seprable_inception_block1(x, activation, regularizer, initializer):

    #### branch 1
    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    #### branch 2
    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(24, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(48, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    #### branch 3
    branch3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    branch3 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    #### branch pool
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)

    branch_pool = Conv2D(16, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)
    
    return concatenate([branch1, branch2, branch3, branch_pool], axis=-1)

def seprable_inception_block2(x, activation, regularizer, initializer):

    #### branch 1
    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    #### branch 2
    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(48, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(96, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    #### branch 3
    branch3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    branch3 = Conv2D(96, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    #### branch pool
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)

    branch_pool = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)
    
    return concatenate([branch1, branch2, branch3, branch_pool], axis=-1)

def seprable_inception_block3(x, activation, regularizer, initializer):

    #### branch 1
    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    branch1 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = activate_layer(branch1, activation)

    #### branch 2
    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    branch2 = Conv2D(96, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = activate_layer(branch2, activation)

    #### branch 3
    branch3 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    branch3 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = activate_layer(branch3, activation)

    #### branch pool
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depthwise_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)

    branch_pool = Conv2D(96, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer)(branch_pool)
    branch_pool = BatchNormalization()(branch_pool)
    branch_pool = activate_layer(branch_pool, activation)
    
    return concatenate([branch1, branch2, branch3, branch_pool], axis=-1)