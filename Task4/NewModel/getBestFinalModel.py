import optuna
from tensorflow.keras import datasets
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import optuna
import pickle
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import to_categorical

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


def create_optuna_model(params):
    backbone = get_best_model()
    
    dense_layers = params["dense_layers"]
    neurons_per_layer = [params[f'n_neurons_layer_{i}'] for i in range(dense_layers)]
    
    regularizer = params["regularizer"]
    lr_r = params["lr_r"]

    optimizer = params["optimizer"]
    lr_o = params["lr_o"]
    
    p = params['Dropout']
    #Build the Multi Layer Perceptron model

    for i in range(dense_layers):
        backbone.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))
        backbone.add(Dropout(p))

    backbone.add(Dense(units=8, activation='softmax',name='output'))
    
    backbone.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    print(backbone.summary())
    
    return backbone

if __name__ == '__main__':
    DATASET_DIR = '../../MIT_small_train_1'
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    BATCH_SIZE = 32
    NUMBER_OF_EPOCHS = 300

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
    loaded_study = optuna.load_study(study_name=name_study, storage="sqlite:///optuna.db")
    print(loaded_study.best_value)
    params = loaded_study.best_params
    print(params)
    
    model = create_optuna_model(params)
    print(model.summary())
    
    """
    history = model.fit(
        train_dataset,
        epochs=NUMBER_OF_EPOCHS,
        validation_data=validation_dataset    
    )
    
    model.save('best_models/model')
    """
    