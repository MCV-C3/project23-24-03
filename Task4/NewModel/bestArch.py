import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import optuna
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

def create_optuna_model(trial):
    
    model = Sequential()

    # CONVOLUTION-------------------------------------------------
    # Optuna suggestions for Conv2D layers and parameters
    i = 0
    filters = trial.suggest_int(f'conv_{i}_filters', 16, 128, log=True)
    kernel_size = trial.suggest_categorical(f'conv_{i}_kernel_size', [3, 5, 7])
    model.add(Conv2D(filters, (kernel_size, kernel_size), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    num_conv = trial.suggest_int('num_conv_layers', 1, 4)
    for i in range(1, num_conv):
        filters = trial.suggest_int(f'conv_{i}_filters', 16, 128, log=True)
        kernel_size = trial.suggest_categorical(f'conv_{i}_kernel_size', [3, 5, 7])
        model.add(Conv2D(filters, (kernel_size, kernel_size), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # CONVOLUTION-------------------------------------------------
    
    # REDUCE DIMS-------------------------------------------------
    
    model.add(Conv2D(10, (1, 1), activation='relu'))
        
    # REDUCE DIMS-------------------------------------------------
    
    # FLATTEN THE CONV--------------------------------------------
    
    model.add(Flatten())
    
    # FLATTEN THE CONV--------------------------------------------
    
    # MLP---------------------------------------------------------

    model.add(Dense(units=CLASSES, activation='softmax',name='output'))
    
    # MLP---------------------------------------------------------

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    return model

def objective(trial):
    model = create_optuna_model_2(trial)
    
    print(model.summary())
        
    history = model.fit(
        train_dataset,
        epochs=NUMBER_OF_EPOCHS,
        validation_data=validation_dataset
    )
    
    return history.history['val_accuracy'][-1]

def create_optuna_model_1(trial):
    
    model = Sequential()

    # CONVOLUTION-------------------------------------------------
    # Optuna suggestions for Conv2D layers and parameters
    i = 0
    filters = trial.suggest_int(f'conv_{i}_filters', 32, 128, log=True)
    kernel_size = trial.suggest_categorical(f'conv_{i}_kernel_size', [3, 5, 7])
    p = trial.suggest_float('Dropout', 0.0, 0.4, step=0.1)
    model.add(Conv2D(filters, (kernel_size, kernel_size), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu', padding='same'))
    model.add(Dropout(p))
    
    num_conv = trial.suggest_int('num_conv_layers', 1, 3)
    for i in range(1, num_conv):
        filters = trial.suggest_int(f'conv_{i}_filters', 32, 128, log=True)
        kernel_size = trial.suggest_categorical(f'conv_{i}_kernel_size', [3, 5, 7])
        model.add(Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same'))
        model.add(Dropout(p))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # CONVOLUTION-------------------------------------------------
    
    # REDUCE DIMS-------------------------------------------------
    if filters > 20:
        new_dim = filters // 2
        model.add(Conv2D(new_dim, (1, 1), activation='relu', name='red_dim'))
        model.add(Dropout(p))
        
    # REDUCE DIMS-------------------------------------------------
    
    # FLATTEN THE CONV--------------------------------------------
    
    model.add(Flatten())
    
    # FLATTEN THE CONV--------------------------------------------
    
    # MLP---------------------------------------------------------

    model.add(Dense(units=CLASSES, activation='softmax',name='output'))
    
    # MLP---------------------------------------------------------

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    return model

def create_optuna_model_2(trial):
    
    model = Sequential()

    # CONVOLUTION-------------------------------------------------
    # Optuna suggestions for Conv2D layers and parameters
    i = 0
    filters = trial.suggest_int(f'conv_{i}_filters', 1, 2, log=True)
    kernel_size = trial.suggest_categorical(f'conv_{i}_kernel_size', [3, 5])
    p = trial.suggest_float('Dropout', 0.0, 0.2, step=0.1)
    model.add(Conv2D(filters, (kernel_size, kernel_size), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu', padding='same'))
    model.add(Dropout(p))
    
    num_conv = trial.suggest_int('num_conv_layers', 1, 2)
    for i in range(1, num_conv):
        filters = trial.suggest_int(f'conv_{i}_filters', 1, 3, log=True)
        kernel_size = trial.suggest_categorical(f'conv_{i}_kernel_size', [3, 5])
        model.add(Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same'))
        model.add(Dropout(p))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # CONVOLUTION-------------------------------------------------
    
    # REDUCE DIMS-------------------------------------------------
    if filters > 10:
        new_dim = 10
        model.add(Conv2D(new_dim, (1, 1), activation='relu', name='red_dim'))
        model.add(Dropout(p))
        
    # REDUCE DIMS-------------------------------------------------
    
    # FLATTEN THE CONV--------------------------------------------
    
    model.add(Flatten())
    
    # FLATTEN THE CONV--------------------------------------------
    
    # MLP---------------------------------------------------------

    model.add(Dense(units=CLASSES, activation='softmax',name='output'))
    
    # MLP---------------------------------------------------------

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    return model



if __name__ == '__main__':

    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    BATCH_SIZE = 16
    NUMBER_OF_EPOCHS = 20
    CLASSES = 100
    
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode="fine")
    
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    datagen = ImageDataGenerator(
        rescale=1./255)
    
    datagen.fit(train_images)
    
    train_dataset = datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE)
    validation_dataset = datagen.flow(test_images, test_labels, batch_size=BATCH_SIZE)
    
    # 1, 2
    name_study = 'bestArch4'
    try:
        optuna.delete_study(storage='sqlite:///optuna.db', study_name=name_study)
    except:
        pass
    
    study = optuna.create_study(storage='sqlite:///optuna.db', study_name=name_study, direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_accuracy = study.best_value

    print(f"Best hyperparameters: {best_params}")
    print(f"Best accuracy: {best_accuracy}")
    

    