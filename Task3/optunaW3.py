import numpy as np
import optuna
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, Flatten
from keras.layers.experimental.preprocessing import Rescaling, RandomFlip, Normalization
from tensorflow.keras.regularizers import l1, l2
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_augmented(dataset_dir, img_height, img_width, batch_size, preprocessing_function):
    train_data_generator = ImageDataGenerator(featurewise_center=False,
                                              samplewise_center=False,
                                              featurewise_std_normalization=False,
                                              samplewise_std_normalization=False,
                                              preprocessing_function=preprocessing_function,
                                              rotation_range=0,
                                              width_shift_range=0.2,
                                              height_shift_range=0.,
                                              shear_range=0.2,
                                              brightness_range=[0.7, 1.3],
                                              zoom_range=0.,
                                              fill_mode='nearest',
                                              horizontal_flip=True,
                                              vertical_flip=False,
                                              rescale=None)

    validation_data_generator = ImageDataGenerator(preprocessing_function=preprocessing_function)

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

def optimizers_config(optimizer, lr):
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'nadam':
        return tf.keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer == 'adadelta':
        return tf.keras.optimizers.Adadelta()

def best_model(optimizer, lr, img_height, img_width):
    base_model = InceptionV3(weights='imagenet', input_shape=(img_height, img_width, 3), include_top=False)

    base_model.trainable = False

    x = base_model.layers[196].output

    x = keras.layers.Conv2D(32, (8, 8),padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Dense(24)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = Dense(8, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr), metrics=['accuracy'])

    for layer in model.layers[165:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
            
    return model

def search_params_optuna(trial, img_height, img_width, preprocessing_function):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    epochs = trial.suggest_categorical("epochs", [20, 50, 80, 100])
    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd', 'nadam', 'adadelta']) 
    learning_rate = trial.suggest_categorical("learning_rate", [0.1, 0.05, 0.01, 0.001, 0.0001])

    result = []
    for i in range(1,5):
        dataset_dir = './MIT_small_train_'+str(i)
        train_dataset, validation_dataset = load_data_augmented(dataset_dir, img_height, img_width, batch_size, preprocessing_function)
        model = best_model(optimizer, learning_rate, img_height, img_width)
        history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=0)
        result.append(history.history['val_accuracy'][-1])
    return np.mean(result)


IMG_HEIGHT=299
IMG_WIDTH = 299
study = optuna.create_study(storage='sqlite:///optunaW3.db', study_name= "task5", direction='maximize')
study.optimize(lambda trial: search_params_optuna(trial, IMG_HEIGHT, IMG_WIDTH, preprocess_input), n_trials=100, n_jobs=1)