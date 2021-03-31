import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
ROUND = -1

def KNN(data_list):
    data, y = data_list[0], data_list[1]
    print(len(np.where(y == 'M')[0]))
    print(len(np.where(y == 'F')[0]))


    print("K NEAREST NEIGHBOR")
    print("********************************************")
    print("")
    k_list = [2,3]
    for i in range(len(data)):

        if ROUND == -1:
            if i == 0:
                print("COMBINED ROUNDS ALL FEATURES UNNORMALIZED")
            elif i ==1:
                print("COMBINED ROUNDS ALL FEATURES NORMALIZED")

        else:
            if i == 0:
                print("ALL FEATURES UNNORMALIZED")
            elif i == 1:
                print("IMAGE FEATURES UNNORMALIZED")
            elif i == 2:
                print("WEIGHT FEATURES UNNORMALIZED")
            elif i == 3:
                print("ALL FEATURES NORMALIZED")
            elif i == 4:
                print("IMAGE FEATURES NORMALIZED")
            elif i == 5:
                print("WEIGHT FEATURES NORMALIZED")


        print("______________________________________________________")
        kf = KFold(n_splits=10)

        X = data[i]
        for k in k_list:

            scores = []
            j = 1

            for train_index, test_index in kf.split(X):

                X_train, X_test = X[train_index], X[test_index]

                if j ==1:
                    print("# training samples:", len(X_train))
                    print("# test samples:", len(X_test))
                    j -= 1

                y_train, y_test = y[train_index], y[test_index]

                neighbors = KNeighborsClassifier(n_neighbors=k)
                neighbors.fit(X_train, y_train)
                predictions = neighbors.predict(X_test)

                acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)

                scores.append(acc)

            print("k =", k  ," | Average accuracy:", np.mean(scores), "(stddev :", np.std(scores) ,")")
            print("")

def getData(filepath):
    data = pd.read_csv(filepath)
    bad_labels = data.loc[(data['label'] != 'M') & (data['label'] != 'F')].index
    data = data.drop(bad_labels, axis=0)

    # split data into training and test sets
    data_training = data
    labels_training = np.array(data_training['label'])

    # format data into numpy arrays
    if ROUND == 3:
        data_training_array = np.column_stack((data_training['length'], data_training['width'],
                                               data_training['L-length'], data_training['T-length'],
                                               data_training['L-width'],
                                               data_training['T-width'], data_training['weight-day1'],
                                               data_training['weight-day2'], data_training['weight-day3'],
                                               data_training['weight-day4'],
                                               data_training['weight-day6']))
    elif ROUND == 4:
        data_training_array = np.column_stack((data_training['length'], data_training['width'],
                                               data_training['L-length'], data_training['T-length'],
                                               data_training['L-width'],
                                               data_training['T-width'], data_training['weight-pre1'], data_training['weight-day1'],
                                               data_training['weight-day2']))


    elif ROUND == 5:
        data_training_array = np.column_stack((data_training['length'], data_training['width'], data_training['L-length'], data_training['T-length'], data_training['L-width'],
                                               data_training['T-width'], data_training['weight-pre1'], data_training['weight-pre2'], data_training['weight-day1'],
                                               data_training['weight-day2'], data_training['weight-day3'], data_training['weight-day4'], data_training['weight-day5'],
                                               data_training['weight-day6']))


    elif ROUND == -1:
        data_training_array = np.column_stack((data_training['length'], data_training['width'],
                                               data_training['L-length'], data_training['T-length'],
                                               data_training['L-width'],
                                               data_training['T-width']))



        # normalize data
        mean = np.mean(data_training_array, axis=0)
        data_training_norm = data_training_array - mean
        std_dev = np.std(data_training_array, axis=0)
        data_training_norm = data_training_norm / std_dev

        train_list = [data_training_array, data_training_norm]

        data_list = [train_list, labels_training]
        return data_list

    data_image_train = data_training_array[:, :6]

    data_weight_train = data_training_array[:, 6:]


    # normalize data
    mean = np.mean(data_training_array, axis=0)
    data_training_norm = data_training_array - mean
    std_dev = np.std(data_training_array, axis=0)
    data_training_norm = data_training_norm / std_dev
    data_image_norm = data_training_norm[:, :6]
    data_weight_norm = data_training_norm[:, 6:]

    train_list = [data_training_array, data_image_train, data_weight_train , data_training_norm, data_image_norm, data_weight_norm]

    data_list = [train_list, labels_training]
    return data_list

def SVM(data_list):
    data, y = data_list[0], data_list[1]

    print("SVM")
    print("********************************************")
    print("")
    for i in range(len(data)):

        if ROUND == -1:
            if i == 0:
                print("COMBINED ROUNDS ALL FEATURES UNNORMALIZED")
            elif i ==1:
                print("COMBINED ROUNDS ALL FEATURES NORMALIZED")

        else:
            if i == 0:
                print("ALL FEATURES UNNORMALIZED")
            elif i == 1:
                print("IMAGE FEATURES UNNORMALIZED")
            elif i == 2:
                print("WEIGHT FEATURES UNNORMALIZED")
            elif i == 3:
                print("ALL FEATURES NORMALIZED")
            elif i == 4:
                print("IMAGE FEATURES NORMALIZED")
            elif i == 5:
                print("WEIGHT FEATURES NORMALIZED")


        print("______________________________________________________")
        kf = KFold(n_splits=10)

        X = data[i]

        print("**********************************************")
        print("")
        print("LINEAR KERNEL")
        print("______________________________________________________")

        scores = []
        j=1

        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]

            if j == 1:
                print("# training samples:", len(X_train))
                print("# test samples:", len(X_test))
                j -= 1

            y_train, y_test = y[train_index], y[test_index]

            if i < 4:
                clf = SVC(kernel='linear', class_weight='balanced')
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                scores.append(acc)

            else:
                clf = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                acc = 1 - (len(y_test) - (y_test == predictions).sum()) /len(y_test)
                scores.append(acc)

        print("Average rate is: ", np.mean(scores))
        print("")
        print("RBF KERNEL")
        print("______________________________________________________")

        gamma_list = [0.0001,0.001,0.01,0.1,1,10,100]

        for gamma in gamma_list:

            scores = []
            j=1

            for train_index, test_index in kf.split(X):

                X_train, X_test = X[train_index], X[test_index]

                if j == 1:
                    print("# training samples:", len(X_train))
                    print("# test samples:", len(X_test))
                    j -= 1

                y_train, y_test = y[train_index], y[test_index]

                if i < 4:
                    clf = SVC(kernel='rbf', gamma=gamma, class_weight='balanced')
                    clf.fit(X_train, y_train)
                    predictions = clf.predict(X_test)
                    acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                    scores.append(acc)

                else:
                    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=gamma, class_weight='balanced'))
                    clf.fit(X_train, y_train)
                    predictions = clf.predict(X_test)
                    acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                    scores.append(acc)

            print("gamma=", gamma," | Average accuracy:", np.mean(scores))

        print("")

def main():
    data_list = getData('../data/measurements/rounds-combined.csv')
    # KNN(data_list)
    SVM(data_list)

if __name__ == '__main__':
    main()
