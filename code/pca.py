import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA


def KPCA(filepath):
    data = pd.read_csv(filepath)
    data_new = np.column_stack((data['length'], data['width'], data['L-length'], data['T-length'], data['L-width'], data['T-width'],data['weight-pre1'], data['weight-day1'], data['weight-day2']))
    transformer = KernelPCA(n_components=2, kernel='rbf', gamma=0.001)
    print(1/len(data['length']))
    data_transformed = transformer.fit_transform(data_new)
    plt.scatter(data_transformed[:,0], data_transformed[:,1], c=data['label'].map({'M': 0, "F": 1}))
    plt.show()

def main():
    KPCA('../data/measurements/round3_formatted_image_paths.csv')
    # example()
if __name__ == '__main__':
    main()
