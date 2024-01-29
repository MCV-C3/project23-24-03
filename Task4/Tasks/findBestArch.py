import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import optuna
from tensorflow.keras.regularizers import l1, l2

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
    kernel_size = trial.suggest_int(f'conv_{i}_kernel_size', 2, 5)
    model.add(Conv2D(filters, (kernel_size, kernel_size), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    num_conv = trial.suggest_int('num_conv_layers', 1, 4)
    for i in range(1, num_conv):
        filters = trial.suggest_int(f'conv_{i}_filters', 16, 128, log=True)
        kernel_size = trial.suggest_int(f'conv_{i}_kernel_size', 2, 5)
        model.add(Conv2D(filters, (kernel_size, kernel_size), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # CONVOLUTION-------------------------------------------------
    
    # FLATTEN THE CONV--------------------------------------------
    
    model.add(Flatten())
    
    # FLATTEN THE CONV--------------------------------------------
    
    # MLP---------------------------------------------------------
        
    dense_layers = trial.suggest_categorical("dense_layers", [1, 2, 3, 4, 5])
    neurons_per_layer = [trial.suggest_categorical(f'n_neurons_layer_{i}', [32, 64, 128, 256, 512]) for i in range(dense_layers)]
    
    regularizer = trial.suggest_categorical("regularizer", [None, 'l2'])
    lr_r = trial.suggest_categorical("lr_r",[0.1, 0.01, 0.001])

    optimizer = trial.suggest_categorical("optimizer", ['adam', 'sgd']) 
    lr_o = trial.suggest_categorical("lr_o", [0.1, 0.01, 0.001])
    
    p = trial.suggest_float('Dropout', 0.0, 0.6, step=0.1)

    
    for i in range(dense_layers):
        model.add(Dense(units=neurons_per_layer[i], activation='relu', kernel_regularizer=regularizers_config(regularizer, lr_r), name=f'dense_layer_{i}'))
        model.add(Dropout(p))

    model.add(Dense(units=8, activation='softmax',name='output'))
    
    # MLP---------------------------------------------------------

    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers_config(optimizer, lr_o), metrics=['accuracy'])
    
    return model

def objective(trial):
    model = create_optuna_model(trial)
    
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    
    history = model.fit(
        train_dataset,
        epochs=NUMBER_OF_EPOCHS,
        validation_data=validation_dataset,
        callbacks=[checkpoint], 
        verbose = 0
    )
    
    return history.history['val_accuracy'][-1]


DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
NUMBER_OF_EPOCHS = 20

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

# study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "bestArch", direction='maximize')
#bestArch <- original
#bestArch <- not runned
study = optuna.create_study(storage='sqlite:///optuna.db', study_name= "bestArch2", direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_accuracy = study.best_value

print(f"Best hyperparameters: {best_params}")
print(f"Best accuracy: {best_accuracy}")