import os
import numpy as np
import random
import shutil

FOLDER_PATH = 'aug'

male_val = []
male_test = []
female_val = []
female_test = []
male_imgs = 0
female_imgs = 0

for sex in os.listdir(FOLDER_PATH):

    num_imgs = len(os.listdir(os.path.join(FOLDER_PATH, sex)))

    if sex == "male":
        male_imgs = num_imgs
        tmp = np.arange(3, male_imgs, 7)
        random.shuffle(tmp)
        stop1 = int(len(tmp) * 0.15)
        stop2 = 2 * stop1
        male_val = tmp[:stop1]
        male_test = tmp[stop1: stop2]

    else:
        female_imgs = num_imgs
        tmp = np.arange(3, female_imgs, 7)
        random.shuffle(tmp)
        stop1 = int(len(tmp) * 0.15)
        stop2 = 2 * stop1
        female_val = tmp[:stop1]
        female_test = tmp[stop1: stop2]

    for file in os.listdir(os.path.join(FOLDER_PATH, sex)):
        image = int(file[:-4])
        image_path = os.path.join(FOLDER_PATH, sex, file)

        if sex == "male":
            if image in male_test:
                save_path = "train_val_test_crop/test/male/" + file
                shutil.copy(image_path, save_path)
            elif image in male_val:
                save_path = "train_val_test_crop/val/male/" + file
                shutil.copy(image_path, save_path)
            else:
                save_path = "train_val_test_crop/train/male/" + file
                shutil.copy(image_path, save_path)

        else:
            if image in female_test:
                save_path = "train_val_test_crop/test/female/" + file
                shutil.copy(image_path, save_path)
            elif image in female_val:
                save_path = "train_val_test_crop/val/female/" + file
                shutil.copy(image_path, save_path)
            else:
                save_path = "train_val_test_crop/train/female/" + file
                shutil.copy(image_path, save_path)




# splitfolders.ratio('aug', output='train_val_test_crop', seed=0, ratio=(0.70, 0.15, 0.15))
