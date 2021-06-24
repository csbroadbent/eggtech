import os
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
import matplotlib.pyplot as plt
from datetime import datetime
from LReLU import LReLU
import os
import numpy as np
import cv2
import random

def create_dataset(folder_path):

    # Load training images
    train_path = folder_path + '/train'

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

def get_data(folder_path, input_shape):
    dim1, dim2, channels = input_shape
    
        # Load training images
    train_path = folder_path + '/train'

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
            img = cv2.resize(img, (dim1, dim2))
            img = np.reshape(dim1, dim2, 1)
            img = np.concat((img, img, img))
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
            img = cv2.resize(img, (dim1, dim2))
            img = np.reshape(dim1, dim2, 1)
            img = np.concat((img, img, img))
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
            img = cv2.resize(img, (dim1, dim2))
            img = np.reshape(dim1, dim2, 1)
            img = np.concat((img, img, img))
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
    