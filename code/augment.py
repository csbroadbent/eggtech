import tensorflow as tf
import numpy as np
import os
import cv2
from matplotlib import pyplot
from skimage.transform import rotate
from skimage.io import imshow, show

split_path = '../data/images/split/train_val_test_crop'

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


x_train, x_val, x_test, y_train, y_val, y_test = create_dataset(split_path)


img_rows, img_cols = 671, 901

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

for image in x_train:
    aug_img = rotate(image, angle=15)
    aug_img = aug_img.reshape(img_rows, img_cols)
    print(aug_img.shape)

    imshow(aug_img)
    show()