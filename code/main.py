import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from datetime import datetime
from LReLU import LReLU
import os
import numpy as np
import cv2
import random

random.seed(552)
split_path = '../data/images/split/train_val_test_length_crop'

def create_dataset(folder_path):

    # Load training images
    train_path = split_path + '/train'

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []

    dir_peek = os.listdir(train_path)[0]
    train_size = len(os.path.join(train_path, dir_peek))
    for dir in os.listdir(train_path):
        for file in os.listdir(os.path.join(train_path, dir)):
            img_path = os.path.join(train_path, dir, file)
            img = cv2.imread(img_path, 0)
            x_train.append(img)
            if dir == 'male':
                y_train.append(0)
            else:
                y_train.append(1)

    # Load test images
    test_path = split_path + '/test'

    for dir in os.listdir(test_path):
        for file in os.listdir(os.path.join(test_path, dir)):
            img_path = os.path.join(test_path, dir, file)
            img = cv2.imread(img_path, 0)
            x_test.append(img)
            if dir == 'male':
                y_test.append(0)
            else:
                y_test.append(1)

    # Load val images
    val_path = split_path + '/val'

    for dir in os.listdir(val_path):
        for file in os.listdir(os.path.join(val_path, dir)):
            img_path = os.path.join(val_path, dir, file)
            img = cv2.imread(img_path, 0)
            x_val.append(img)
            if dir == 'male':
                y_val.append(0)
            else:
                y_val.append(1)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_val = np.array(x_val)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    return x_train, x_val, x_test, y_train, y_val, y_test

# load and normalize data
x_train, x_val, x_test, y_train, y_val, y_test = create_dataset(split_path)

train = list(zip(x_train, y_train))
test = list(zip(x_test, y_test))
val = list(zip(x_val, y_val))
random.shuffle(train)
random.shuffle(test)
random.shuffle(val)

x_train, y_train = zip(*train)
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test, y_test = zip(*test)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_val, y_val = zip(*val)
x_val = np.array(x_val)
y_val = np.array(y_val)


batch_size = 1
num_classes = 2
epochs = 20

# input image dimensions
img_rows, img_cols = 1600, 1200


# convert data format to channel last format
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# structure 1
# model.add(Conv2D(1, kernel_size=(4, 4),
#                 activation='linear',
#                 input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Activation(LReLU))
# model.add(Dense(num_classes, activation='softmax'))

# structure 2
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='linear',
                 input_shape=input_shape,
                 padding='same'))
model.add(BatchNormalization())
model.add(Activation(LReLU))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same'))
model.add(BatchNormalization())
model.add(Activation(LReLU))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

logdir = "./logs/HW4_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[tensorboard_callback],
          )

score = model.evaluate(x_test, y_test, verbose=0)

print('Test accuracy:', score[1])
