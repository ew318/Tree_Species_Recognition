import tensorflow as tf
import os
from models.data_setup import data_generators
import models.mymodels_configurable as mymodelsconfig
import pickle
from results.generate_metrics import get_metrics

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
        model = mymodelsconfig.vgg6(IMG_DIM, classesj, 'adamax', 'he_normal')
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        model.fit(
            train_data_genj,
            epochs=epochs,
            validation_data=val_data_genj,
            callbacks=callbacks)
        eval = model.evaluate(test_data_genj, verbose=0)
        # store prediction stats
        results = get_metrics(model, test_data_genj)
        results['accuracy'] = eval[1]
        results['loss'] = eval[0]
        with open('best_model/model' + str(classesj) + '.txt', 'wb') as file_pi:
            pickle.dump(results, file_pi)


vary_class_number()
