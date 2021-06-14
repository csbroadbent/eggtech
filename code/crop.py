import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

folder_path = '../data/images/split/combined_length'

width_max = 0
height_max = 0

crop_coords_male = []
crop_coords_female = []

i = 0
j = 0
num_pics = 0
for sex in os.listdir(folder_path):
    for file in os.listdir(os.path.join(folder_path, sex)):
        num_pics += 1

        inputImage = cv2.imread(os.path.join(folder_path, sex, file), 0)
        inputImageGray = cv2.bitwise_not(inputImage)

        edges = cv2.Canny(inputImageGray, 300, 400)

        mask = np.ones((1200,1600)).astype(int)
        col_mask = np.zeros((1200,50))
        row_mask = np.zeros((50,1600))

        mask[:50] = row_mask
        mask[-50:] = row_mask
        mask[:,:50] = col_mask
        mask[:,-50:] = col_mask

        edges = edges / 255
        edges = edges.astype(int)

        edges = np.bitwise_and(edges, mask)
        edges = edges * 255
        edges = edges.astype(float)

        top = 0
        right = 0
        bottom = 0
        left = 0

        # find max row of top part of egg
        for i in range(1200):
            if np.sum(edges[i]) != 0:
                top = i
                break

        # find min row of bottom part of egg
        for i in range(1199, 0, -1):
            if np.sum(edges[i]) != 0:
                bottom = i
                break

        # # find min col of left part of egg
        for i in range(1600):
            if np.sum(edges[:,i]) != 0:
                left = i
                break

        # # find max col of top part of egg
        for i in range(1599, 0, -1):
            if np.sum(edges[:,i]) != 0:
                right = i
                break

        edges[top] = np.ones(1600)
        edges[bottom] = np.ones(1600)
        edges[:,left] = np.ones(1200)
        edges[:,right] = np.ones(1200)

        width = right - left
        height = bottom - top

        if width > width_max:
            width_max = width

        if height > height_max:
            height_max = height

        if sex == 'female':
            crop_coords_female.append([left, right, top, bottom])
            i += 1
        if sex == 'male':
            crop_coords_male.append([left, right, top, bottom])
            j += 1

print("Cropped egg dimensions should be: ", width_max, " X ", height_max)

img_dim = max(width_max, height_max)

print("num pics: ", num_pics)
i = 0
male_path = folder_path + '/male'
for file in os.listdir(os.path.join(male_path)):

    left = crop_coords_male[i][0]
    right = crop_coords_male[i][1]
    top = crop_coords_male[i][2]
    bottom = crop_coords_male[i][3]

    img = cv2.imread(os.path.join(male_path, file), 0)

    img = img[top - 15:, left - 25:]
    new_width = right - left
    new_height = bottom - top
    img = img[:new_height + 25, :new_width + 45]
    h, w = img.shape

    sq_img = np.ones((img_dim + 50, img_dim + 50)) * 255
    hh, ww = sq_img.shape

    yoff = round((hh - h) / 2)
    xoff = round((ww - w) / 2)

    # if i >= 93:
    #     rand_pixels = img.flatten()
    #     rand_pixels = rand_pixels[rand_pixels > 175]
    #     sq_img = np.random.choice(rand_pixels, size=(hh,ww))

    sq_img[yoff:yoff+h, xoff:xoff+w] = img

    edges = cv2.Canny(np.uint8(sq_img), 250, 450)
    thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(edges, cnts, [255, 255, 255])

    sq_img_bin = edges / 255

    sq_img = np.multiply(sq_img, sq_img_bin)
    sq_img = (255 - (sq_img_bin * 255)) + sq_img



    img_name = '../data/images/split/cropped_length/male/' + str(i) + '.bmp'
    cv2.imwrite(img_name, sq_img)

    i += 1

i = 0
female_path = folder_path + '/female'
for file in os.listdir(os.path.join(female_path)):

    left = crop_coords_female[i][0]
    right = crop_coords_female[i][1]
    top = crop_coords_female[i][2]
    bottom = crop_coords_female[i][3]

    img = cv2.imread(os.path.join(female_path, file), 0)

    img = img[top - 15:, left - 25:]
    new_width = right - left
    new_height = bottom - top
    img = img[:new_height + 25, :new_width + 45]
    h, w = img.shape

    sq_img = np.ones((img_dim + 50, img_dim + 50)) * 255
    hh, ww = sq_img.shape

    yoff = round((hh - h) / 2)
    xoff = round((ww - w) / 2)

    sq_img[yoff:yoff + h, xoff:xoff + w] = img

    edges = cv2.Canny(np.uint8(sq_img), 250, 450)
    thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(edges, cnts, [255, 255, 255])

    sq_img_bin = edges / 255

    sq_img = np.multiply(sq_img, sq_img_bin)
    sq_img = (255 - (sq_img_bin * 255)) + sq_img

    img_name = '../data/images/split/cropped_length/female/' + str(i) + '.bmp'
    cv2.imwrite(img_name, sq_img)
    i+= 1

