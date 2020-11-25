import tensorflow as tf
import os
from models.data_setup import data_generators
from models.vgg16 import vgg16_base, vgg16_dropout, vgg16_weights

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

IMG_DIM = os.getenv('IMG_DIM', 150)
epochs = os.getenv('EPOCHS', 100)
batch_size = os.getenv('BATCH_SIZE', 64)
i = 78
classes, train_data_gen, val_data_gen, test_data_gen = data_generators(i, batch_size=batch_size, IMG_DIM=IMG_DIM)

# TRAIN MODELS
vgg16_base(train_data_gen, val_data_gen, test_data_gen, num_classes=classes, IMG_DIM=IMG_DIM, epochs=epochs)
vgg16_weights(train_data_gen, val_data_gen, test_data_gen, num_classes=classes, IMG_DIM=IMG_DIM, epochs=epochs)
vgg16_dropout(train_data_gen, val_data_gen, test_data_gen, num_classes=classes, IMG_DIM=IMG_DIM, epochs=epochs, dropout=0.1)
vgg16_dropout(train_data_gen, val_data_gen, test_data_gen, num_classes=classes, IMG_DIM=IMG_DIM, epochs=epochs, dropout=0.2)


# TRAIN 'BEST' MODEL FOR VARYING NUMBERS OF SPECIES
def vary_class_number(model):
    for j in range(1, 79):
        classesj, train_data_genj, val_data_genj, test_data_genj = data_generators(j)
        model(train_data_genj, val_data_genj, test_data_genj, num_classes=classesj, IMG_DIM=IMG_DIM, epochs=epochs)


#vary_class_number(vgg16_base)
