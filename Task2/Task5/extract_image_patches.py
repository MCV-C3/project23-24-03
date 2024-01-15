import tensorflow as tf
from utils import *

def generate_image_patches_db(in_directories, out_directory, im_resize=None, patch_size=64):
    contador = 0
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    
    for in_dir in os.listdir(in_directories):
        
        out_in_folders = os.path.join(out_directory, in_dir)
        if not os.path.exists(out_in_folders):
                os.makedirs(out_in_folders)
        
        actual_folder_dir = os.path.join(in_directories, in_dir)
        for class_folder in os.listdir(actual_folder_dir):
            
            out_folders = os.path.join(out_in_folders, class_folder)
            if not os.path.exists(out_folders):
                os.makedirs(out_folders)
            
            actual_folder_class_dir = os.path.join(actual_folder_dir, class_folder)
            for image_path in os.listdir(actual_folder_class_dir):
                
                actual_image_file = os.path.join(actual_folder_class_dir, image_path)
                
                if im_resize is not None:
                    original_image = Image.open(actual_image_file).resize((im_resize, im_resize)).convert('RGB')
                else:
                    original_image = Image.open(actual_image_file).convert('RGB')
                    
                image_array = np.array(original_image)
                height, width, channels = image_array.shape
                
                height_step = patch_size
                width_step = patch_size
                
                index = 0
                for i in range(0, height, height_step):
                    for j in range(0, width, width_step):
                        actual_split = image_array[i : i + height_step, j : j + width_step, :]
                        image_split = Image.fromarray(actual_split)
                        
                        image_name = os.path.join(out_folders, image_path.split('.')[0] + '_' + str(index) + '.' + image_path.split('.')[1])
                        
                        print(image_name)
                        image_split.save(image_name)
                        index += 1
                        
                        
if __name__ == '__main__':
    input_folder = os.path.join('..', 'MIT_split')
    output_folder = 'patches'
    
    generate_image_patches_db(input_folder, output_folder, 128, 32)

