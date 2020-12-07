from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D


def vgg5(IMG_DIM, num_classes, opt, ki):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=ki, padding='same', input_shape=(IMG_DIM, IMG_DIM, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=ki))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def vgg6(IMG_DIM, num_classes, opt, ki):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=ki, padding='same', input_shape=(IMG_DIM, IMG_DIM, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(1024, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(Conv2D(1024, (3, 3), activation='relu', kernel_initializer=ki, padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=ki))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
