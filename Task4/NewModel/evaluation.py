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

if __name__ == '__main__':
    model = tf.keras.models.load_model('best_models/model_CIFAR')
    
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    BATCH_SIZE = 8
    NUMBER_OF_EPOCHS = 100
    CLASSES = 100
    
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode="fine")
    
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    datagen = ImageDataGenerator(
        rescale=1./255)
    
    datagen.fit(train_images)
    
    train_dataset = datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE)
    validation_dataset = datagen.flow(test_images, test_labels, batch_size=BATCH_SIZE)
    
    print(model.evaluate(test_images, test_labels))