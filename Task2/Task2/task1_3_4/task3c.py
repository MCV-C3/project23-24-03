import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image

for patch_s in [50, 80, 120, 160, 200]:
    for eval in [voting, softmax_mean]:
        #user defined variables
        AGGREGATION = eval
        PATCH_SIZE  = 64
        PATCHES = 40
        PATCHES_TEST = patch_s
        BATCH_SIZE  = 16
        HIDDEN_LAYERS = 5
        NEURONS_PER_LAYER = [4096, 2048, 1024, 256, 256]
        OPTIMIZER = 'sgd'
        LEARNING_RATE = 0.01
        EPOCHS = 150
        DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
        PATCHES_DIR = '/ghome/group03/development/C3/data/MIT_split_patches_'+str(PATCH_SIZE)+"_"+str(PATCHES)
        MODEL_FNAME = '/ghome/group03/development/C3/patch_based_mlp_'+str(PATCH_SIZE)+'_'+str(PATCHES)+'.weights.h5'

        if not os.path.exists(DATASET_DIR):
            print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
            quit()
        if not os.path.exists(PATCHES_DIR):
            print('WARNING: patches dataset directory '+PATCHES_DIR+' does not exist!\n')
            print('Creating image patches dataset into '+PATCHES_DIR+'\n')
            generate_image_patches_db(DATASET_DIR,PATCHES_DIR,patch_size=PATCH_SIZE, max_patches = PATCHES)
            print('patxes generated!\n')

        train_dataset, validation_dataset = load_data_patch(PATCHES_DIR, PATCH_SIZE, BATCH_SIZE, preprocessing_method = 'rescaling')

        model = build_model_patch(PATCH_SIZE, HIDDEN_LAYERS, NEURONS_PER_LAYER, None, None, OPTIMIZER, LEARNING_RATE, 'train')

        train = False
        if  not os.path.exists(MODEL_FNAME) or train:
            
            history = model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, verbose=0)
            
            model.save_weights(MODEL_FNAME)  # always save your weights after training or during training

        print('Building MLP model for testing...\n')

        model = build_model_patch(PATCH_SIZE, HIDDEN_LAYERS, NEURONS_PER_LAYER, None, None, OPTIMIZER, LEARNING_RATE, 'test')

        model.load_weights(MODEL_FNAME)

        directory = DATASET_DIR+'/test'
        classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
        correct = 0.
        total   = 807
        count   = 0

        for class_dir in os.listdir(directory):
            cls = classes[class_dir]
            for imname in os.listdir(os.path.join(directory,class_dir)):
              im = Image.open(os.path.join(directory,class_dir,imname))
              patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=PATCHES_TEST)
              out = model.predict(patches/255.)
              predicted_cls = AGGREGATION(out)
              if predicted_cls == cls:
                correct+=1
              count += 1
              print('Evaluated images: '+str(count)+' / '+str(total), end='\r')
            
        print('Done c!\n')
        print(f'PATCH_SIZE {PATCH_SIZE} PATCHES {PATCHES} PATCHES TEST {PATCHES_TEST} AGGREGATION {AGGREGATION.__name__}')
        print('Test Acc. = '+str(correct/total)+'\n')