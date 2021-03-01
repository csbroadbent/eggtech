import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from mpl_toolkits import mplot3d
from scipy import stats

def create2d_plots(filepath):

    data = np.genfromtxt(filepath, delimiter=',')
    data = np.delete(data, (0), axis=0)
    data = np.delete(data, (3), axis=1)
    data = np.delete(data, (0), axis=1)

    labels = pd.read_csv(filepath)['Label'].values

    for i in range(len(labels) - 1, -1, -1):
        if labels[i] == 'M':
            labels[i] = 0
        elif labels[i] == 'F':
            labels[i] = 1
        else:
            data = np.delete(data, (i), axis=0)
            labels = np.delete(labels, (i), axis=0)

    color_dict = {0:'blue', 1:'red'}
    shape_dict = {3:'o', 4:'^', 5:'+'}

    # min_index = 0
    # min_width = 1000
    # for i in range(len(labels)):
    #     if data[min_index][0] < min_width:
    #         min_index = i
    #         min_width = data[min_index][0]
    #
    # print(min_width)
    #
    # data = np.delete(data, (min_index), axis=0)
    # labels = np.delete(labels, (min_index), axis=0)

    for i in range(len(labels)):
        plt.scatter(data[i][0], data[i][1], c=color_dict[labels[i]], label=labels[i], marker=shape_dict[data[i][2]])

    plt.legend(labels=['male','female'],)
    plt.xlabel("length (mm)")
    plt.ylabel("width (mm")
    plt.title('Rounds 3,4, and 5')
    plt.savefig('rounds-combined.png')
    plt.show()

def create3d_plots(filepath):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    data = np.genfromtxt(filepath, delimiter=',')
    data = np.delete(data, (0), axis=0)
    data = np.delete(data, (3), axis=1)
    data = np.delete(data, (0), axis=1)


    labels = pd.read_csv(filepath)['Label'].values

    for i in range(len(labels) - 1, -1, -1):
        if labels[i] == 'M':
            labels[i] = 0
        elif labels[i] == 'F':
            labels[i] = 1
        else:
            data = np.delete(data, (i), axis=0)
            labels = np.delete(labels, (i), axis=0)

    ratios = np.array([len(data)])
    ratios = data[:,1]/data[:,0]

    color_dict = {0: 'blue', 1: 'red'}

    for i in range(len(labels)):
        ax.scatter3D(data[i][0], data[i][1], ratios[i], c=color_dict[labels[i]], label=labels[i])

    ax.legend(labels=['male','female'])
    ax.set_xlabel("length (mm)")
    ax.set_ylabel("width (mm")
    ax.set_zlabel("ratio w/l")
    plt.show()

def t_test(filepath):

    data = np.genfromtxt(filepath, delimiter=',')
    data = np.delete(data, (0), axis=0)

    labels = pd.read_csv(filepath)['label'].values

    male = []
    female = []

    for i in range(len(labels) - 1, -1, -1):
        if labels[i] == 'M':
            labels[i] = 0
            male.append(data[i])
        elif labels[i] == 'F':
            labels[i] = 1
            female.append(data[i])

    male = np.asarray(male)
    female = np.asarray(female)

    print("Total # male samples: ", len(male), " | Total # of female samples: ", len(female))
    print("Performed t-test on length: p-value = ", stats.ttest_ind(male[:,1], female[:,1])[1])
    print("Performed t-test on  width: p-value = ", stats.ttest_ind(male[:,3], female[:,3])[1])
    print("Performed t-test on  ratios: p-value = ", stats.ttest_ind(male[:,1]/male[:,3], female[:,1]/female[:,3])[1])
    print("Performed t-test on L-length: p-value = ", stats.ttest_ind(male[:, 6], female[:, 6])[1])
    print("Performed t-test on T-length: p-value = ", stats.ttest_ind(male[:, 7], female[:, 7])[1])
    print("Performed t-test on A-length: p-value = ", stats.ttest_ind(male[:, 8], female[:, 8])[1])
    print("Performed t-test on E-length: p-value = ", stats.ttest_ind(male[:, 9], female[:, 9])[1])
    print("Performed t-test on L-width: p-value = ", stats.ttest_ind(male[:, 10], female[:, 10])[1])
    print("Performed t-test on T-width: p-value = ", stats.ttest_ind(male[:, 11], female[:, 11])[1])
    print("Performed t-test on A-width: p-value = ", stats.ttest_ind(male[:, 12], female[:, 12])[1])
    print("Performed t-test on E-width: p-value = ", stats.ttest_ind(male[:, 13], female[:, 13])[1])





def main():
    # create2d_plots("../data/measurements/rounds-combined.csv")
    # create3d_plots("../data/measurements/round4_formatted.csv")
    t_test("../data/measurements/round3_formatted_image_paths.csv")

if __name__ == '__main__':
    main()

