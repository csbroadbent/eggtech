import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os

NO_LABELS = False


def combine_rounds(data_list, day):
    r3X, r3Y = data_list[0][0], data_list[0][1]
    r4X, r4Y = data_list[1][0], data_list[1][1]
    r5X, r5Y = data_list[2][0], data_list[2][1]

    X = np.concatenate((r3X, r4X, r5X), axis=0)
    y = np.concatenate((r3Y, r4Y, r5Y), axis=0)

    data = np.insert(X, 0, y, axis=1)

    savepath = "../data/voc/labeled/combined/Day" + str(day) + "-combined.csv"
    np.savetxt(savepath, data, delimiter=',')

def get_data_matrix(filepath):

    if NO_LABELS == True:
        data = np.genfromtxt(filepath, delimiter=',')

        y = data[:,0].astype(int)
        X = data[:,1:]

        return X, y

    df = pd.read_csv(filepath)

    invalid_labels = df.loc[df['label'] == -1].index
    df = df.drop(invalid_labels, axis=0)  # remove -1 labels

    y = np.array(df['label']).astype(int)

    df = df.drop(['Data range', 'label'], axis=1)
    df = df.drop(df.columns[0], axis=1)

    X = np.array(df)


    return X, y

def KNN(folderpath):
    for filename in os.listdir(folderpath):
        if filename[-4:] != '.csv': #skip over folders
            continue

        filepath = folderpath + '/' + filename
        X, y = get_data_matrix(filepath)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        print(filename[:-4])
        print("K NEAREST NEIGHBOR")
        print("********************************************")
        print("")

        k_list = [2, 3]

        kf = KFold(n_splits=10)

        for train_index, test_index in kf.split(X):  # inelegant way to print train and test size
            print("Training size:", len(train_index))
            print("Testing size:", len(test_index))
            break

        print("# males:", len(np.where(y == 0)[0]))
        print("# females:", len(np.where(y == 1)[0]))

        for k in k_list:

            scores = []

            for train_index, test_index in kf.split(X):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # scale data
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                neighbors = KNeighborsClassifier(n_neighbors=k)
                neighbors.fit(X_train, y_train)
                predictions = neighbors.predict(X_test)

                acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)

                scores.append(acc)

            print("k =", k  ," | Average Accuracy:", np.mean(scores))

        print("")

def SVM(folderpath):
    for filename in os.listdir(folderpath):
        if filename[-4:] != '.csv': #skip over folders
            continue

        filepath = folderpath + '/' + filename
        X, y = get_data_matrix(filepath)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        print(filename[:-4])
        print("SVM")
        print("********************************************")
        print("")

        kf = KFold(n_splits=5)

        for train_index, test_index in kf.split(X):  # inelegant way to print train and test size
            print("Training size:", len(train_index))
            print("Testing size:", len(test_index))
            break

        print("# males:", len(np.where(y == 0)[0]))
        print("# females:", len(np.where(y == 1)[0]))
        print("")
        print("LINEAR KERNEL")
        print("______________________________________________________")

        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
            scores.append(acc)

        print("Average accuracy is: ", np.mean(scores))
        print("")
        print("RBF KERNEL")
        print("______________________________________________________")

        gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]

        for gamma in gamma_list:

            scores = []

            for train_index, test_index in kf.split(X):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=gamma, class_weight='balanced'))
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                scores.append(acc)

            print("gamma=", gamma, " | Average accuracy:", np.mean(scores))

        print("")


def combine():
    for i in range(1, 7):
        if i == 5:
            continue

        r3filepath = '../data/voc/labeled/round3/R3-Day' + str(i) + '.csv'
        r4filepath = '../data/voc/labeled/round4/Day ' + str(i) + '.csv'
        r5filepath = '../data/voc/labeled/round5/Day ' + str(i) + '.csv'

        r3 = [[], []]
        r4 = [[], []]
        r5 = [[], []]

        r3[0], r3[1] = get_data_matrix(r3filepath)

        r4[0], r4[1] = get_data_matrix(r4filepath)
        r5[0], r5[1] = get_data_matrix(r5filepath)
        data_list = [r3,r4,r5]
        combine_rounds(data_list, i)

def main():
    folderpath = '../data/voc/labeled/combined'
    # KNN(folderpath)
    # SVM(folderpath)

    combine()





if __name__ == '__main__':
    main()