import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
import math
import numpy as np


# MODEL DEFINITIONS
def vgg16_base(train_data_gen, val_data_gen, test_data_gen, num_classes=78, IMG_DIM=150, epochs=100):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_DIM, IMG_DIM, 3))
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Implement early stopping - stop training if no improvement in accuracy for 5 consecutive epochs
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint('vgg16_models/base_' + str(num_classes),
                                                    save_best_only=True, monitor='val_loss')]

    history = model.fit(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
        callbacks=callbacks)

    with open('vgg16_models/base_history_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    scores = model.evaluate(test_data_gen, verbose=0)
    with open('vgg16_models/base_evaluation_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(scores, file_pi)

    del scores, model, history, vgg_model


def vgg16_dropout(train_data_gen, val_data_gen, test_data_gen, num_classes=78, IMG_DIM=150, epochs=100, dropout=0.2):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_DIM, IMG_DIM, 3))
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Implement early stopping - stop training if no improvement in accuracy for 5 consecutive epochs
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint('vgg16_models/dropout' + str(int(100*dropout)) + '_' + str(num_classes),
                                                    save_best_only=True, monitor='val_loss')]

    history = model.fit(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
        callbacks=callbacks)

    with open('vgg16_models/dropout' + str(int(100*dropout)) + '_history_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    scores = model.evaluate(test_data_gen, verbose=0)
    with open('vgg16_models/dropout' + str(int(100*dropout)) + '_evaluation_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(scores, file_pi)

    del scores, model, history, vgg_model


def vgg16_weights(train_data_gen, val_data_gen, test_data_gen, num_classes=78, IMG_DIM=150, epochs=100, batch_size=64):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_DIM, IMG_DIM, 3))
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Implement early stopping - stop training if no improvement in accuracy for 5 consecutive epochs
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint('vgg16_models/weights_' + str(num_classes),
                                                    save_best_only=True, monitor='val_loss')]
    # Find class weights
    num_examples = len(train_data_gen.filenames)
    number_of_generator_calls = math.ceil(num_examples / (1 * batch_size))
    train_labels = []
    for k in range(0, int(number_of_generator_calls)):
        train_labels.extend(np.array(train_data_gen[k][1]))

    labels_numerated = []
    for j in range(0, len(train_labels)):
        for k in range(0, num_classes):
            if train_labels[j][k]:
                labels_numerated.append(k)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels_numerated), labels_numerated)

    weights = {}
    for j in range(0, len(class_weights)):
        weights[j] = class_weights[j]

    history = model.fit(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen, class_weight=weights,
        callbacks=callbacks)

    with open('vgg16_models/weights_history_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    scores = model.evaluate(test_data_gen, verbose=0)
    with open('vgg16_models/weights_evaluation_' + str(num_classes), 'wb') as file_pi:
        pickle.dump(scores, file_pi)

    del scores, model, history, vgg_model
