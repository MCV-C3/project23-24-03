import tensorflow as tf
from utils import *
import pickle

if __name__ == '__main__':
    class_labels = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
    model = tf.keras.models.load_model('bestModel/task2Best')
    print(model.summary())
    
    layer_name = 'dense_layer_2'
    output_name = 'output'
    intermediate_layer_model = tf.keras.Model(inputs=model.input, 
                                              outputs=model.get_layer(layer_name).output)
    
    output_layer_model = tf.keras.Model(inputs=model.input,
                                        outputs=model.get_layer(output_name).output)
    
    #user defined variables
    IMG_SIZE    = 32
    BATCH_SIZE  = 1
    DATASET_DIR = 'patches'

    train_dataset, validation_dataset = load_data(DATASET_DIR, IMG_SIZE, BATCH_SIZE, preprocessing_method = 'rescaling', shuffle=False)
    
    train_data = []
    train_labels = []
    
    image_list = []
    image_label = []
    for data, label in train_dataset:
        image_list.append(intermediate_layer_model(data).numpy().squeeze())
        
        if len(image_list) == 16:
            train_data.append(image_list.copy())
            train_labels.append(class_labels[np.argmax(label.numpy())])
            
            image_list = []
        
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    with open('train_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f)
    
    test_data = []
    test_labels = []
    
    image_list = []
    image_label = []
    for data, label in validation_dataset:
        
        image_list.append(intermediate_layer_model(data).numpy().squeeze())
        
        if len(image_list) == 16:
            test_data.append(image_list.copy())
            test_labels.append(class_labels[np.argmax(label.numpy())])
            
            image_list = []
        
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
        
    with open('test_labels.pkl', 'wb') as f:
        pickle.dump(test_labels, f)

    