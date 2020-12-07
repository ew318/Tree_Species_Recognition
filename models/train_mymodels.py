import tensorflow as tf
import os
import pickle
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from models.data_setup import data_generators
import models.mymodels as mymodels
import models.mymodels_configurable as mymodelsconfig

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


IMG_DIM = os.getenv('IMG_DIM', 150)
epochs = os.getenv('EPOCHS', 100)
batch_size = os.getenv('BATCH_SIZE', 64)
i = 6
num_classes, train_data_gen, val_data_gen, test_data_gen = data_generators(i, batch_size=batch_size, IMG_DIM=IMG_DIM)


def fit_evaluate_model(model, result_folder, model_name, epochs=100):
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(
                     result_folder + '/' + model_name + '_' + str(num_classes), save_best_only=True, monitor='val_loss')]
    history = model.fit(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
        callbacks=callbacks)

    # store model history
    with open(result_folder + '/' + model_name + '_history_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # store evaluation stats
    with open(result_folder + '/' + model_name + '_evaluation_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(model.evaluate(test_data_gen, verbose=0), file_pi)

    # generate and save confusion matrix
    result = model.predict(test_data_gen)
    result_fix = []
    for item in result:
        result_fix.append([1 if y == max(item) else 0 for y in item])

    test_labels = []
    for j in range(0, int(math.ceil(len(test_data_gen.filenames) / (1 * 64)))):
        test_labels.extend(np.array(test_data_gen[j][1]))

    predictions = []
    labels = []
    for j in range(0, len(test_labels)):
        labels.append(list(test_labels[j]).index(1))
        predictions.append(result_fix[j].index(1))

    # store confusion matrix
    with open(result_folder + '/' + model_name + '_cm_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(confusion_matrix(labels, predictions), file_pi)

    del model, history, predictions, labels, result, test_labels, result_fix


optimizers = {
    'sgd': SGD(lr=0.001, momentum=0.9),
    'adam': Adam(),
    'adamax': Adamax(),
    'adadelta': Adadelta(),
    'adagrad': Adagrad(),
    'ftrl': Ftrl(),
    'nadam': Nadam(),
    'rmsprop': RMSprop()
}

initializers = [
    'constant', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal',
    'lecun_uniform', 'random_normal', 'truncated_normal'
]

# Try lots of different small models, see what works best
fit_evaluate_model(mymodels.get_simple_cnn(IMG_DIM, num_classes), 'mymodel', 'baseline')
fit_evaluate_model(mymodels.vgg1(IMG_DIM, num_classes), 'mymodel', 'vgg1')
fit_evaluate_model(mymodels.vgg2(IMG_DIM, num_classes), 'mymodel', 'vgg2')
fit_evaluate_model(mymodels.vgg3(IMG_DIM, num_classes), 'mymodel', 'vgg3')
fit_evaluate_model(mymodels.vgg4(IMG_DIM, num_classes), 'mymodel', 'vgg4')
fit_evaluate_model(mymodels.vgg5(IMG_DIM, num_classes), 'mymodel', 'vgg5')
fit_evaluate_model(mymodels.vgg3_dropout_20(IMG_DIM, num_classes), 'mymodel', 'vgg3_dropout_20')
fit_evaluate_model(mymodels.vgg3_dropout_30(IMG_DIM, num_classes), 'mymodel', 'vgg3_dropout_30')
fit_evaluate_model(mymodels.vgg3_weight_decay(IMG_DIM, num_classes), 'mymodel', 'vgg3_weight_decay')
fit_evaluate_model(mymodels.vgg3_dropout_batchnorm(IMG_DIM, num_classes), 'mymodel', 'vgg3_dropout_batchnorm')
fit_evaluate_model(mymodels.vgg3_kernel5(IMG_DIM, num_classes), 'mymodel', 'vgg3_kernel5')
fit_evaluate_model(mymodels.vgg3_opt_adam(IMG_DIM, num_classes), 'mymodel', 'vgg3_opt_adam')
fit_evaluate_model(mymodels.vgg3_ki_none(IMG_DIM, num_classes), 'mymodel', 'vgg3_ki_none')

# Generalise models, try with multiple paramater combinations.
for initializer in initializers:
    for optimizer in optimizers:
        fit_evaluate_model(mymodelsconfig.vgg5(IMG_DIM, num_classes, optimizer, initializer),
                           'mymodel_gen', 'vgg5_' + optimizer + '_' + initializer)
        fit_evaluate_model(mymodelsconfig.vgg6(IMG_DIM, num_classes, optimizer, initializer),
                           'mymodel_gen', 'vgg6_' + optimizer + '_' + initializer)
