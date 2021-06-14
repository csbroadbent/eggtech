import numpy as np
import os
import cv2
from matplotlib import pyplot
from skimage.transform import rotate
from skimage.io import imshow, show

split_path = '../data/images/split/cropped_length'
save_path = '../data/images/split/aug'

def create_dataset(folder_path):

    # Load training images

    male_img = []
    female_img = []

    for dir in os.listdir(folder_path):
        for file in os.listdir(os.path.join(folder_path, dir)):
            img_path = os.path.join(folder_path, dir, file)
            img = cv2.imread(img_path, 0)
            if dir == 'male':
                male_img.append(img)
            else:
                female_img.append(img)

    male_img = np.array(male_img)
    female_img = np.array(female_img)

    return male_img, female_img


male_img, female_img = create_dataset(split_path)

angles = [-30, -20, -10, 0, 10, 20, 30]
img_list = [male_img, female_img]
for i in range(len(img_list)):

    j = 0

    if i == 0:
        img_path = save_path + '/male/'
    else:
        img_path = save_path + '/female/'

    for image in img_list[i]:
        rows = cols = image.shape[0]
        for angle in angles:
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            aug_img = cv2.warpAffine(image, M, (cols, rows),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))

            path = img_path + str(j) + '.bmp'
            cv2.imwrite(path, aug_img)

            j += 1


