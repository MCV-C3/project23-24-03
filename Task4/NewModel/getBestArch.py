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

def create_optuna_model(params):
    
    model = Sequential()

    # CONVOLUTION-------------------------------------------------
    # Optuna suggestions for Conv2D layers and parameters
    
    filters = params[f'conv_0_filters']
    kernel_size = params[f'conv_0_kernel_size']
    model.add(Conv2D(filters, (kernel_size, kernel_size), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    num_conv = params['num_conv_layers']
    for i in range(1, num_conv):
        filters = params[f'conv_{i}_filters']
        kernel_size = params[f'conv_{i}_kernel_size']
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
    
    optimizer = params["optimizer"] 
    lr_o = params["lr_o"]

    model.add(Dense(units=CLASSES, activation='softmax',name='output'))
    
    # MLP---------------------------------------------------------

    optimizer = params['optimizer']
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    return model

def create_optuna_model_1(params):
    
    model = Sequential()
    p = params['Dropout']
    # CONVOLUTION-------------------------------------------------
    # Optuna suggestions for Conv2D layers and parameters
    
    filters = params[f'conv_0_filters']
    kernel_size = params[f'conv_0_kernel_size']
    model.add(Conv2D(filters, (kernel_size, kernel_size), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'))
    model.add(Dropout(p))
    
    num_conv = params['num_conv_layers']
    for i in range(1, num_conv):
        filters = params[f'conv_{i}_filters']
        kernel_size = params[f'conv_{i}_kernel_size']
        model.add(Conv2D(filters, (kernel_size, kernel_size), activation='relu'))
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

    optimizer = params["optimizer"] 
    lr_o = params["lr_o"]
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    return model



if __name__ == '__main__':
    loaded_study = optuna.load_study(study_name="bestArch3", storage="sqlite:///optuna.db")
    print(loaded_study.best_value)
    params = loaded_study.best_params
    print(params)
    
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    BATCH_SIZE = 16
    NUMBER_OF_EPOCHS = 200
    CLASSES = 8
    DATASET_DIR = '../../MIT_small_train_1'
    
    
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
    
    
    model = create_optuna_model_1(params)
    
    print(model.summary())
    
    history = model.fit(
        train_dataset,
        epochs=NUMBER_OF_EPOCHS,
        validation_data=validation_dataset
    )
    
    print(history.history['val_accuracy'][-1])
    model.save('best_models/model_small_1')
    