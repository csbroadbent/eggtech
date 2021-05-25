import csv
import os
import sys
import time
import pandas as pd
import psutil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy
from shutil import copyfile

def write_path(folderpath, filepath):

    new_filepath = filepath[0:-4] + '_image_paths.csv'

    data = pd.read_csv(filepath)

    image_path_list = []

    for image_path in os.listdir(folderpath):
        image_path_list.append(image_path)

    with open(new_filepath, 'w') as writefile:
        csvwriter = csv.writer(writefile, delimiter=',', lineterminator='\n')
        csvwriter.writerow(['ID', 'length', 'imagepath-length', 'width', 'imagepath-width', 'label'])  # write in column names
        for i in range(len(image_path_list)//2):
            csvwriter.writerow([data['ID'][i], data['length'][i], image_path_list[2*i], data['width'][i], image_path_list[2*i + 1], data['Label'][i]])

    return

def move_images(folderpath, filepath):
    data = pd.read_csv(filepath)
    folder_dst = '../data/images/split/round4/length/'

    for i in range(len(data)):
        if data['label'][i] == 'M':
            src = folderpath + data['imagepath-length'][i]
            dst = folder_dst + "male-length-4/" + data['imagepath-length'][i]
            copyfile(src,dst)

        elif data['label'][i] == 'F':
            src = folderpath + data['imagepath-length'][i]
            dst = folder_dst + "female-length-4/" + data['imagepath-length'][i]
            copyfile(src, dst)

def check_images(folderpath, filepath):

    data = pd.read_csv(filepath)
    i = 0

    for image_path in os.listdir(folderpath):
        img = Image.open(folderpath + "/" + image_path)

        id = str(int(data['ID'][i // 2]))

        img = img.rotate(180)

        img_draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('sans-serif.ttf', 60)
        img_draw.text((750, 1000), id, fill='green', font=font)

        process_list = []
        for proc in psutil.process_iter():
            process_list.append(proc)

        img.show()
        time.sleep(1)
        for proc in psutil.process_iter():
            if not proc in process_list:
                proc.kill()

        i += 1


def main():
    folderpath = "../data/images/round4/"
    filepath = "../data/measurements/round-4.csv"

    move_images(folderpath, filepath)


if __name__ == '__main__':
    main()

