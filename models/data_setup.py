import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Get classes to be used each time
camden_all = pd.read_csv('/scratch/emiwat01/projects/camden/data/camden_clean.csv')
counts = camden_all.groupby('Common Name').count()[['Number Of Trees']].sort_values(
    'Number Of Trees', axis=0, ascending=False, inplace=False, kind='quicksort',
    na_position='last', ignore_index=False, key=None)


# Function to build dataframe of image data
def build_df(type, which_species):
    data_dict = {'filename': [],
                 'class': []
                 }
    camden_path = os.getenv('IMAGE_DIR', '/scratch/emiwat01/projects/camden/data/camden_split_images_no_logo/')
    camden_path = camden_path + type
    for subdir, dirs, files in os.walk(camden_path):
        for file in files:
            if len(subdir.split('/')) == 9:
                species = subdir.split('/')[8]
                if species in which_species:
                    data_dict['filename'].append(camden_path + '/' + species + '/' + file)
                    data_dict['class'].append(species)
    return pd.DataFrame(data_dict, columns=['filename', 'class'])


def data_generators(i=6, batch_size=64, IMG_DIM=150):
    considered_species = list(counts.index[:i])
    train_dataframe = build_df('train', considered_species)
    validation_dataframe = build_df('validation', considered_species)
    test_dataframe = build_df('test', considered_species)
    num_classes = len(train_dataframe['class'].unique())
    # Augment training images
    train_image_generator = ImageDataGenerator(
        rescale=1. / 255, rotation_range=45,
        width_shift_range=.15, height_shift_range=.15,
        horizontal_flip=True, zoom_range=0.5, brightness_range=[0.8, 1.2])
    # Generator for our validation and test data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    # Image Generators
    train_data_gen = train_image_generator.flow_from_dataframe(
        train_dataframe, directory=None, x_col='filename', y_col='class', target_size=(IMG_DIM, IMG_DIM),
        class_mode='categorical', batch_size=batch_size, shuffle=True)
    val_data_gen = validation_image_generator.flow_from_dataframe(
        validation_dataframe, directory=None, x_col='filename', y_col='class', target_size=(IMG_DIM, IMG_DIM),
        class_mode='categorical', batch_size=batch_size, shuffle=True)
    test_data_gen = validation_image_generator.flow_from_dataframe(
        test_dataframe, directory=None, x_col='filename', y_col='class', target_size=(IMG_DIM, IMG_DIM),
        class_mode='categorical', batch_size=batch_size, shuffle=True)
    return num_classes, train_data_gen, val_data_gen, test_data_gen
