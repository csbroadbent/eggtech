import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
import math
from sklearn.model_selection import KFold
import re
pd.options.mode.chained_assignment = None  # default='warn'
FILE_NAME = 'R3-Day8.csv'
FOLDER_PATH = '../data/voc/labeled/combined/'
LABELS = False

def t_test(filepath):
    df = pd.read_csv(filepath)
    print(df)

def label_voc(voc_path, label_path):

    xl = pd.ExcelFile(voc_path)
    egg_data = pd.read_csv(label_path)
    sheet_names = xl.sheet_names


    for sheet in sheet_names:

        day = xl.parse(sheet)
        day.insert(1, 'label',-1)

        for name in day['Data range']:
            id_split = name.split('_')

            if len(id_split) > 2:

                if (id_split[1][:2] == 'ID') and (len(id_split[1]) == 6):
                    index = day.loc[day['Data range'] == name].index[0]
                    print(id_split)
                    id = int(id_split[1])
                    egg = egg_data.loc[egg_data['ID'] == id]

                    if len(egg) == 0:
                        day['label'][index] = -1
                    elif egg['label'].values[0] == 'M':
                        day['label'][index] = 0
                    elif egg['label'].values[0] == 'F':
                        day['label'][index] = 1
                    else:
                        day['label'][index] = -1

        save_path = '../data/voc/labeled/round5/' + sheet + '.csv'


def My_PCA(filepath, filename):
    data = pd.read_csv(filepath, sep=',', index_col=False)

    if LABELS == False:
        data = data.to_numpy()
        labels = data[:,0].astype(int)
        data = np.delete(data, 0, axis=1)
        invalid_label = np.where(labels == -1)
        data_matrix = np.delete(data, invalid_label, axis=0)

    else:
        invalid_label = data.loc[data['label'] == -1].index

        data_drop = data.drop(invalid_label, axis=0)
        labels = data_drop['label'].values
        data_drop = data_drop.drop(['Data range', 'label'], axis=1)
        data_matrix = data_drop.values
        data_matrix = np.delete(data_matrix, 0, axis=-1)

    var_list = [0.90, 0.95, 0.99]

    for var in var_list:

        pca = PCA(n_components=var, svd_solver='full')
        pca.fit(data_matrix)
        data_transformed = pca.transform(data_matrix)

        data_transformed = np.insert(data_transformed, 0, labels.astype(int), axis=1)

        file_name = '../data/voc/labeled/combined/PCA/var=' + str(var) + '/' + filename[:-4] + '-PCA-var=' + str(var) + '.csv'
        np.savetxt(file_name, data_transformed, delimiter=',')

def KPCA(filepath, filename):

    data = pd.read_csv(filepath, sep=',', index_col=False)


    if LABELS == False:
        data = data.to_numpy()
        labels = data[:,0].astype(int)
        data = np.delete(data, 0, axis=1)
        invalid_label = np.where(labels == -1)
        data_matrix = np.delete(data, invalid_label, axis=0)

    else:
        invalid_label = data.loc[data['label'] == -1].index
        data_drop = data.drop(invalid_label, axis=0)
        labels = data_drop['label'].values
        data_drop = data_drop.drop(['Data range', 'label'], axis=1)

        data_matrix = data_drop.values
        data_matrix = np.delete(data_matrix, 0, axis=-1)

    mean = np.mean(data_matrix, axis=0)
    data_norm = data_matrix - mean
    std_dev = np.std(data_matrix, axis=0)

    zero_stddev = np.where(std_dev == 0)

    std_dev = np.delete(std_dev, zero_stddev, axis=0)
    data_norm = np.delete(data_norm, zero_stddev, axis=1)
    data_norm = data_norm / std_dev

    gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]
    for gamma in gamma_list:

        transformer = KernelPCA(kernel='rbf', gamma=gamma)
        data_transformed = transformer.fit_transform(data_norm)
        data_transformed = np.insert(data_transformed, 0, labels.astype(int), axis=1)
        gamma_path = 'gamma=' + str(gamma) +'/'
        file_name = '../data/voc/labeled/combined/KPCA/gamma=' + str(gamma) + '/' + filename[:-4] + '-KPCA-rbf-gamma=' + str(gamma) + '.csv'
        np.savetxt(file_name, data_transformed, delimiter=',')

    # deg_list = [2,3,4]
    # for deg in deg_list:
    #     transformer = KernelPCA(kernel='poly', degree=deg)
    #     data_transformed = transformer.fit_transform(data_norm)
    #     data_transformed = np.insert(data_transformed, 0, labels.astype(int), axis=1)
    #     deg_path = 'deg=' + str(deg) + '/'
    #     file_name = '../data/voc/labeled/combined/KPCA/poly/deg=' + str(deg) + '/' + filename[:-4] + '-KPCA-poly-deg=' + str(deg) + '.csv'
    #     np.savetxt(file_name, data_transformed, delimiter=',')

def create_KPCA_plots(folderpath):

    for folder in os.listdir(folderpath):

        path = folderpath + '/' + folder

        for filename in os.listdir(path):

            if filename[-4:] != '.csv':
                continue

            filepath = path + '/' + filename

            data = np.genfromtxt(filepath, delimiter=',')
            labels = data[:,0].astype(int)
            data = data[:,1:]

            colors = ['blue' if l == 0 else 'red' for l in labels]

            plt.scatter(data[:, 1], data[:, 2], c=colors)

            blue_patch = mpatches.Patch(color='blue', label='Male')
            red_patch = mpatches.Patch(color='red', label='Female')
            plt.legend(handles=[blue_patch, red_patch])


            savepath = path + '/plots/' + filename + '.png'
            plt.savefig(savepath)
            plt.close()

def create_PCA_plots(folderpath):

    for filename in os.listdir(folderpath):

        if filename[-4:] != '.csv':
            continue

        filepath = folderpath + '/' + filename

        data = np.genfromtxt(filepath, delimiter=',')
        labels = data[:,0].astype(int)
        data = data[:,1:]

        colors = ['blue' if l == 0 else 'red' for l in labels]

        plt.scatter(data[:, 1], data[:, 2], c=colors)

        blue_patch = mpatches.Patch(color='blue', label='Male')
        red_patch = mpatches.Patch(color='red', label='Female')
        plt.legend(handles=[blue_patch, red_patch])

        savepath = folderpath[:-3] + 'PCA_plots/' + filename + '.png'
        plt.savefig(savepath)
        plt.close()

def get_diff(day1_path, day2_path):

    day1 = pd.read_csv(day1_path)
    day2 = pd.read_csv(day2_path)

    invalid_label = day1.loc[day1['label'] == -1].index

    day1 = day1.drop(invalid_label, axis=0)
    day2 = day2.drop(invalid_label, axis=0)

    labels = day1['label'].values
    data_drop1 = day1.drop(['Data range', 'label'], axis=1)
    data_drop2 = day2.drop(['Data range', 'label'], axis=1)
    data_matrix = data_drop1.values - data_drop2.values
    data_matrix = np.delete(data_matrix, 0, axis=1)
    data_matrix = np.insert(data_matrix, 0, labels, axis=1)

    file_name = '../data/voc/labeled/round5/diff/Days1-2-diff.csv'

    np.savetxt(file_name, data_matrix, delimiter=',')

    data_matrix = np.delete(data_matrix, 0, axis=1)

    transformer = KernelPCA(kernel='linear')
    data_transformed = transformer.fit_transform(data_matrix)
    data_transformed = np.insert(data_transformed, 0, labels.astype(int), axis=1)

    file_name = '../data/voc/labeled/round5/PCA/Days1-2-PCA.csv'
    np.savetxt(file_name, data_transformed, delimiter=',')

    mean = np.mean(data_matrix, axis=0)
    data_norm = data_matrix - mean
    std_dev = np.std(data_matrix, axis=0)

    zero_stddev = np.where(std_dev == 0)

    std_dev = np.delete(std_dev, zero_stddev, axis=0)
    data_norm = np.delete(data_norm, zero_stddev, axis=1)
    data_norm = data_norm / std_dev

    gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]
    for gamma in gamma_list:

        transformer = KernelPCA(kernel='rbf', gamma=gamma)
        data_transformed = transformer.fit_transform(data_norm)
        data_transformed = np.insert(data_transformed, 0, labels.astype(int), axis=1)
        gamma_path = 'gamma=' + str(gamma) +'/'
        file_name = '../data/voc/labeled/round5/KPCA/gamma=' + str(gamma) + '/Days1-2-KPCA-rbf-gamma=' + str(gamma) + '.csv'
        np.savetxt(file_name, data_transformed, delimiter=',')

def combine_day_pairs(folderpath, days):

    day1_path = folderpath + 'Day ' + str(days[0]) + '.csv'
    day2_path = folderpath + 'Day ' + str(days[1]) + '.csv'

    day1_df = pd.read_csv(day1_path)
    day2_df = pd.read_csv(day2_path)

    drop_indices = []

    for index, id in zip(day1_df.index,day1_df['Data range']):
        reg = re.search("(\d{4})", id)
        if not reg:
            drop_indices.append(index)
        else:
            new_id = reg.group()
            day1_df.at[index, 'Data range'] = new_id

    day1_df = day1_df.drop(labels=drop_indices)

    drop_indices = []

    for index, id in zip(day2_df.index,day2_df['Data range']):
        reg = re.search("(\d{4})", id)
        if not reg:
            drop_indices.append(index)
        else:
            new_id = reg.group()
            day2_df.at[index, 'Data range'] = new_id

    day2_df = day2_df.drop(labels=drop_indices)

    day1_ordered_indices = []
    day2_ordered_indices = []

    day2_df = day2_df.drop(columns=['label'], axis=1)
    day2_df = day2_df.drop(columns=day2_df.columns[[0]], axis=1)

    merged = pd.merge(day1_df, day2_df, how='inner', on='Data range')
    merged = merged.drop(columns=merged.columns[[0]], axis=1)
    merged = merged.drop(merged.loc[merged['label']==-1].index)

    filename = 'Days' + str(days[0]) + '&' + str(days[1]) + '.csv'
    savepath = folderpath + filename

    merged.to_csv(savepath, index=False)




def main():
    # voc_path = '../data/voc/r5-uniform.xlsx'
    # label_path = '../data/measurements/round5_complete.csv'
    # label_voc(voc_path, label_path)
    #
    # for filename in os.listdir(FOLDER_PATH):
    #     if filename[-4:] == '.csv':
    #         filepath = '../data/voc/labeled/combined/' + filename
    #         My_PCA(filepath, filename)

    # create_KPCA_plots(FOLDER_PATH)

    # for filename in os.listdir('../data/voc/labeled/round5'):
    #     if filename[-4:] == '.csv':
    #         path = '../data/voc/labeled/round5/' + filename
    #         PCA(path, filename)

    # day1_path = '../data/voc/labeled/round5/diff/Day 1.csv'
    # day2_path = '../data/voc/labeled/round5/diff/Day 2.csv'
    # get_diff(day1_path, day2_path)

    # t_test('../data/voc/labeled/combined/day6/Day6-combined.csv')
    days_list = [[1,5], [1,6], [2,5], [2,6]]
    for days in days_list:
        combine_day_pairs('../data/voc/labeled/round3/', days)



if __name__ == '__main__':
    main()