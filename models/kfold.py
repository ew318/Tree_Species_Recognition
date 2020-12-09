import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import tensorflow as tf
import os
import pickle
import models.mymodels_configurable as mymodelsconfig
from results.generate_metrics import get_metrics

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

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


def data_generators(model, model_name, opt, ki, i=6, batch_size=64, IMG_DIM=150):
    considered_species = list(counts.index[:i])
    train_dataframe = build_df('train', considered_species)
    validation_dataframe = build_df('validation', considered_species)
    test_dataframe = build_df('test', considered_species)
    # Merge all data together
    frames = [train_dataframe, validation_dataframe, test_dataframe]
    all_data = pd.concat(frames)
    # Create results dict to store metrics in
    results = {
        'accuracy': [],
        'avg_precision': [],
        'avg_recall': [],
    }
    # Now split into 5 folds
    kfold = KFold(5, shuffle=True, random_state=1)
    for train, test in kfold.split(all_data):
        train_data = all_data[all_data.index.isin(train)]
        test_data = all_data[all_data.index.isin(test)]
        # Augment training images, resize generators for valid + test data
        train_image_generator = ImageDataGenerator(
            validation_split=0.2, rescale=1. / 255, rotation_range=45,
            width_shift_range=.15, height_shift_range=.15,
            horizontal_flip=True, zoom_range=0.5, brightness_range=[0.8, 1.2])
        valid_image_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
        test_image_generator = ImageDataGenerator(rescale=1. / 255)
        # Split image data sets
        train_data_gen = train_image_generator.flow_from_dataframe(
            train_data, directory=None, x_col='filename', y_col='class', target_size=(IMG_DIM, IMG_DIM),
            class_mode='categorical', batch_size=batch_size, shuffle=True, subset='training')
        val_data_gen = valid_image_generator.flow_from_dataframe(
            train_data, directory=None, x_col='filename', y_col='class', target_size=(IMG_DIM, IMG_DIM),
            class_mode='categorical', batch_size=batch_size, shuffle=True, subset='validation')
        test_data_gen = test_image_generator.flow_from_dataframe(
            test_data, directory=None, x_col='filename', y_col='class', target_size=(IMG_DIM, IMG_DIM),
            class_mode='categorical', batch_size=batch_size, shuffle=True)
        # Now train and evaluate kth model
        mymodel = model(IMG_DIM=IMG_DIM, num_classes=i, opt=opt, ki=ki)
        accuracy, precision, recall = train_k_model(mymodel, train_data_gen, val_data_gen, test_data_gen)
        results['accuracy'].append(accuracy)
        results['avg_precision'].append(precision)
        results['avg_recall'].append(recall)
        del mymodel
    # store results
    with open('kfold/' + model_name + '.txt', 'wb') as file_pi:
        pickle.dump(results, file_pi)


def train_k_model(model, train_data_gen, val_data_gen, test_data_gen):
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    model.fit(
        train_data_gen,
        epochs=100,
        validation_data=val_data_gen,
        callbacks=callbacks)
    # store evaluation stats
    eval = model.evaluate(test_data_gen, verbose=0)[1]
    # store prediction stats
    predict = get_metrics(model, test_data_gen)
    return eval, predict['avg_precision'], predict['avg_recall']


data_generators(model=mymodelsconfig.vgg6,
                model_name='6_adamax_he_normal',
                opt='adamax',
                ki='he_normal')

data_generators(model=mymodelsconfig.vgg6,
                model_name='6_adamax_lecun_normal',
                opt='adamax',
                ki='lecun_normal')

data_generators(model=mymodelsconfig.vgg5,
                model_name='5_adamax_glorot_normal',
                opt='adamax',
                ki='glorot_normal')

data_generators(model=mymodelsconfig.vgg6,
                model_name='5_adamax_truncated_normal',
                opt='adamax',
                ki='truncated_normal')

data_generators(model=mymodelsconfig.vgg6,
                model_name='6_adamax_he_uniform',
                opt='adamax',
                ki='he_uniform')

