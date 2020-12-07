import tensorflow as tf
import os
from models.data_setup import data_generators
import models.mymodels_configurable as mymodelsconfig
from models.train_mymodels import fit_evaluate_model

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

IMG_DIM = os.getenv('IMG_DIM', 150)
epochs = os.getenv('EPOCHS', 100)
batch_size = os.getenv('BATCH_SIZE', 64)


# TRAIN 'BEST' MODEL FOR VARYING NUMBERS OF SPECIES
def vary_class_number():
    for j in range(1, 79):
        classesj, train_data_genj, val_data_genj, test_data_genj = data_generators(j, batch_size=batch_size,
                                                                                   IMG_DIM=IMG_DIM)
        model = mymodelsconfig.vgg6(IMG_DIM, j, 'adamax', 'he_normal')

        fit_evaluate_model(model, 'best_model', 'best_' + str(j))
        model(train_data_genj, val_data_genj, test_data_genj, num_classes=classesj, IMG_DIM=IMG_DIM, epochs=epochs)


vary_class_number()
