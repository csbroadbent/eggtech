import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA


def KPCA(filepath):
    data = pd.read_csv(filepath)
    data_new = np.column_stack((data['length'], data['width'], data['L-length'], data['T-length'], data['L-width'], data['T-width'],data['weight-pre1'], data['weight-pre2'],data['weight-day1'], data['weight-day2'], data['weight-day3'], data['weight-day4'], data['weight-day5'], data['weight-day6']))
    mean = np.mean(data_new, axis=0)
    data_new = data_new - mean
    std_dev = np.std(data_new, axis=0)
    data_new = data_new / std_dev
    data_image = data_new[:, :6]
    data_weight = data_new[:, 6:]


    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.001)
    data_transformed = transformer.fit_transform(data_new)
    plt.scatter(data_transformed[:,0], data_transformed[:,1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.001 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.001.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.01)
    data_transformed = transformer.fit_transform(data_new)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.01 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.01.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.1)
    data_transformed = transformer.fit_transform(data_new)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.1 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.1.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=1)
    data_transformed = transformer.fit_transform(data_new)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=1 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=1.png')
    plt.close()

    # weights
    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.001)
    data_transformed = transformer.fit_transform(data_weight)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.001 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.001-weight.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.01)
    data_transformed = transformer.fit_transform(data_weight)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.01 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.01-weight.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.1)
    data_transformed = transformer.fit_transform(data_weight)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.1 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.1-weight.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=1)
    data_transformed = transformer.fit_transform(data_weight)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=1 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=1-weight.png')
    plt.close()

    # image features
    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.001)
    data_transformed = transformer.fit_transform(data_image)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.001 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.001-image.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.01)
    data_transformed = transformer.fit_transform(data_image)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.01 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.01-image.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=0.1)
    data_transformed = transformer.fit_transform(data_image)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=0.1 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=0.1-image.png')
    plt.close()

    transformer = KernelPCA(n_components=None, kernel='rbf', gamma=1)
    data_transformed = transformer.fit_transform(data_image)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c=data['label'].map({'M': 0, "F": 1}))
    plt.title('Rbf gamma=1 weight only')
    plt.savefig('../results/pca/round5/rbf/gamma=1-image.png')
    plt.close()




def main():
    KPCA('../data/measurements/round5_complete.csv')
    # example()
if __name__ == '__main__':
    main()
