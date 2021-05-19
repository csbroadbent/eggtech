import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from mpl_toolkits import mplot3d
from scipy import stats

def create_lambda_tau_plots(filepath):
    data = pd.read_csv(filepath)

    round3_male = data[(data['label'] == 'M') & (data['round'] == 3)]
    round3_female = data[(data['label'] == 'F') & (data['round'] == 3)]

    round4_male = data[(data['label'] == 'M') & (data['round'] == 4)]
    round4_female = data[(data['label'] == 'F') & (data['round'] == 4)]

    round5_male = data[(data['label'] == 'M') & (data['round'] == 5)]
    round5_female = data[(data['label'] == 'F') & (data['round'] == 5)]

    color_dict = {0: 'blue', 1: 'red'}
    shape_dict = {3: 'o', 4: '^', 5: '+'}

    plt.scatter(round3_male['L-length'], round3_male['T-length'], c='blue')
    plt.scatter(round3_female['L-length'], round3_female['T-length'], c='red')
    plt.legend(labels=['male','female'],)
    plt.xlabel("length-lambda")
    plt.ylabel("length-tau")
    plt.title('Rounds 3 Length Lambda-Tau')
    plt.savefig('round3-length-lambda-tau.png')
    plt.close()

    plt.scatter(round3_male['L-width'], round3_male['T-width'], c='blue')
    plt.scatter(round3_female['L-width'], round3_female['T-width'], c='red')
    plt.legend(labels=['male','female'],)
    plt.xlabel("width-lambda")
    plt.ylabel("width-tau")
    plt.title('Rounds 3 Width Lambda-Tau')
    plt.savefig('round3-width-lambda-tau.png')
    plt.close()

    plt.scatter(round4_male['L-length'], round4_male['T-length'], c='blue')
    plt.scatter(round4_female['L-length'], round4_female['T-length'], c='red')
    plt.legend(labels=['male','female'],)
    plt.xlabel("length-lambda")
    plt.ylabel("length-tau")
    plt.title('Rounds 4 Length Lambda-Tau')
    plt.savefig('round4-length-lambda-tau.png')
    plt.close()

    plt.scatter(round4_male['L-width'], round4_male['T-width'], c='blue')
    plt.scatter(round4_female['L-width'], round4_female['T-width'], c='red')
    plt.legend(labels=['male','female'],)
    plt.xlabel("width-lambda")
    plt.ylabel("width-tau")
    plt.title('Rounds 4 Width Lambda-Tau')
    plt.savefig('round4-width-lambda-tau.png')
    plt.close()

    plt.scatter(round5_male['L-length'], round5_male['T-length'], c='blue')
    plt.scatter(round5_female['L-length'], round5_female['T-length'], c='red')
    plt.legend(labels=['male','female'],)
    plt.xlabel("length-lambda")
    plt.ylabel("length-tau")
    plt.title('Rounds 5 Length Lambda-Tau')
    plt.savefig('round5-length-lambda-tau.png')
    plt.close()

    plt.scatter(round5_male['L-width'], round5_male['T-width'], c='blue')
    plt.scatter(round5_female['L-width'], round5_female['T-width'], c='red')
    plt.legend(labels=['male','female'],)
    plt.xlabel("width-lambda")
    plt.ylabel("width-tau")
    plt.title('Rounds 5 Width Lambda-Tau')
    plt.savefig('round5-width-lambda-tau.png')
    plt.close()

    plt.scatter(round3_male['L-length'], round3_male['T-length'], c='blue', marker='^')
    plt.scatter(round3_female['L-length'], round3_female['T-length'], c='red', marker='^')
    plt.scatter(round4_male['L-length'], round4_male['T-length'], c='cyan', marker='+')
    plt.scatter(round4_female['L-length'], round4_female['T-length'], c='magenta', marker='+')
    plt.scatter(round5_male['L-length'], round5_male['T-length'], c='green', marker='o')
    plt.scatter(round5_female['L-length'], round5_female['T-length'], c='yellow', marker='o')
    plt.legend(labels=['round3 male','round3 female', 'round4 male', 'round4 female', 'round5 male', 'round5 female'],)
    plt.xlabel("length-lambda")
    plt.ylabel("length-tau")
    plt.title('Combined rounds Length Lambda-Tau')
    plt.savefig('combined-length-lambda-tau.png')
    plt.close()

    plt.scatter(round3_male['L-width'], round3_male['T-width'], c='blue', marker='^')
    plt.scatter(round3_female['L-width'], round3_female['T-width'], c='red', marker='^')
    plt.scatter(round4_male['L-width'], round4_male['T-width'], c='cyan', marker='+')
    plt.scatter(round4_female['L-width'], round4_female['T-width'], c='magenta', marker='+')
    plt.scatter(round5_male['L-width'], round5_male['T-width'], c='green', marker='o')
    plt.scatter(round5_female['L-width'], round5_female['T-width'], c='yellow', marker='o')
    plt.legend(labels=['round3 male','round3 female', 'round4 male', 'round4 female', 'round5 male', 'round5 female'],)
    plt.xlabel("width-lambda")
    plt.ylabel("width-tau")
    plt.title('Combined rounds Width Lambda-Tau')
    plt.savefig('combined-width-lambda-tau.png')
    plt.close()

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
    # t_test("../data/measurements/round5_formatted_image_paths.csv")
    create_lambda_tau_plots('../data/measurements/rounds-combined.csv')
if __name__ == '__main__':
    main()

