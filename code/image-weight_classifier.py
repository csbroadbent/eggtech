import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import KernelPCA
from scipy.spatial import distance_matrix
from scipy.special import expit
from tqdm import tqdm
ROUND = 3

def LR(data_list):
    data, y = data_list[0], data_list[1]

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

        scores = []
        j = 1

        roc_avg = 0

        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]

            if j == 1:
                print("# training samples:", len(X_train))
                print("# test samples:", len(X_test))
                j -= 1

            y_train, y_test = y[train_index], y[test_index]

            lr = LogisticRegression(max_iter=250)
            lr.fit(X_train, y_train)
            predictions = lr.predict(X_test)

            acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)

            scores.append(acc)
            roc_avg += roc_auc_score(y_test, predictions)

        print("Average accuracy:", np.mean(scores))
        print("")


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
        kf = StratifiedKFold(n_splits=5)

        X = data[i]
        for k in k_list:

            scores = []
            j = 1

            for train_index, test_index in kf.split(X, y):

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
    labels_training = pd.factorize(data_training['label'])[0]
    print("males: ", len(data_training[data_training['label'] == 'M']))
    print("females: ", len(data_training[data_training['label'] == 'F']))
    # format data into numpy arrays
    if ROUND == 3 or ROUND == '3&5':
        data_training_array = np.column_stack((data_training['length'], data_training['width'],
                                               data_training['L-length'], data_training['T-length'],
                                               data_training['L-width'],
                                               data_training['T-width'], data_training['weight-day1'],
                                               data_training['weight-day2'], data_training['weight-day3'],
                                               data_training['weight-day4'],data_training['weight-day5'],
                                               data_training['weight-day6']))
    elif ROUND == 4:
        labels_training = 1 - labels_training
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


    elif ROUND == 'all':
        data_training_array = np.column_stack((data_training['length'], data_training['width'],
                                               data_training['L-length'], data_training['T-length'],
                                               data_training['L-width'],
                                               data_training['T-width'], data_training['weight-day1'], data_training['weight-day2']))



        # # normalize data
        # # normalize data
        # mean = np.mean(data_training_array, axis=0)
        # data_training_norm = data_training_array - mean
        # std_dev = np.std(data_training_array, axis=0)
        # data_training_norm = data_training_norm / std_dev
        # data_image_norm = data_training_norm[:, :6]
        # data_weight_norm = data_training_norm[:, 6:]
        #
        # train_list = [data_training_array, data_image_train, data_weight_train, data_training_norm, data_image_norm,
        #               data_weight_norm]
        #
        # data_list = [train_list, labels_training]
        #
        # return data_list

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
    # table = pd.DataFrame(columns=['C', 'gamma', 'avg. Accuracy', 'std.', 'avg. AUROC', 'std.']) # for single gamma
    table = pd.DataFrame(
        columns=['C', 'gamma_img', 'avg. Accuracy', 'std.', 'avg. AUROC', 'std.'])  # for different gammas

    data_info = pd.DataFrame(columns=['# training samples', '# test samples', '# males', '# females', 'male/fem ratio'])
    data, y = data_list[0], data_list[1]
    round_name = ''

    print("# males:", len(np.where(y == 0)[0]))
    print("# females:", len(np.where(y == 1)[0]))

    print("SVM")
    print("********************************************")
    print("")
    for i in range(len(data)):

        if ROUND == -1:
            if i == 0:
                print("COMBINED ROUNDS ALL FEATURES UNNORMALIZED")
                round_name = "COMBINED ROUNDS ALL FEATURES UNNORMALIZED"
            elif i == 1:
                print("COMBINED ROUNDS ALL FEATURES NORMALIZED")
                round_name = "COMBINED ROUNDS ALL FEATURES NORMALIZED"

        else:
            if i == 0:
                print("ALL FEATURES UNNORMALIZED")
                round_name = "ALL FEATURES UNNORMALIZED"
            elif i == 1:
                print("IMAGE FEATURES UNNORMALIZED")
                round_name = "IMAGE FEATURES UNNORMALIZED"
            elif i == 2:
                print("WEIGHT FEATURES UNNORMALIZED")
                round_name = "WEIGHT FEATURES UNNORMALIZED"
            elif i == 3:
                print("ALL FEATURES NORMALIZED")
                round_name = "ALL FEATURES NORMALIZED"
            elif i == 4:
                print("IMAGE FEATURES NORMALIZED")
                round_name = "IMAGE FEATURES NORMALIZED"
            elif i == 5:
                print("WEIGHT FEATURES NORMALIZED")
                round_name = "WEIGHT FEATURES NORMALIZED"


        print("______________________________________________________")
        kf = StratifiedKFold(n_splits=10)

        X = data[i]

        scores = []
        j=1

        print("")
        print("RBF KERNEL")
        print("______________________________________________________")


        j = 1


        gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]
        C_list = [0.001, 0.01, 0.1, 1, 10, 100]

        max_auroc = (0, 0, 0, 0, 0, 0)


        for gamma in tqdm(gamma_list):
            for C in C_list:

                scores = []
                roc_scores = []

                for train_index, test_index in kf.split(X, y):

                    X_train, X_test = X[train_index], X[test_index]

                    if j == 1:
                        # print("# training samples:", len(X_train))
                        # print("# test samples:", len(X_test))
                        # print("# males:", len(np.where(y == 0)[0]))
                        # print("# females:", len(np.where(y == 1)[0]))
                        # print("male/fem ratio:", (len(np.where(y == 0)[0]) / len(y)))
                        data_list = [len(X_train), len(X_test), len(np.where(y == 0)[0]), len(np.where(y == 1)[0]), (len(np.where(y == 0)[0]) / len(y))]
                        j -= 1

                        if ROUND == 'all':
                            round_num = 'All rounds combined'
                        else:
                            round_num = 'Round ' + str(ROUND)

                        data_info.loc[round_num] = data_list

                    y_train, y_test = y[train_index], y[test_index]



                    if i < 3:
                        clf = make_pipeline(
                                            SVC(kernel='rbf', gamma=gamma, class_weight='balanced', probability=True))
                        clf.fit(X_train, y_train)
                        predictions = clf.predict(X_test)
                        acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                        scores.append(acc)

                        w = clf.decision_function(X_test)
                        w_zip = list(zip(w, y_test, predictions))
                        w_zip.sort(key=lambda x: x[0])
                        w_zip.reverse()
                        w, y_test, predictions = zip(*w_zip)
                        w = np.array(w)

                        y_test = np.array(y_test)
                        predictions = np.array(predictions)
                        acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                        scores.append(acc)

                        roc_scores.append(roc_auc_score(y_test, w))



                    else:
                        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=gamma, class_weight='balanced', probability=True))
                        clf.fit(X_train, y_train)
                        predictions = clf.predict(X_test)
                        acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                        scores.append(acc)

                        w = clf.decision_function(X_test)
                        w_zip = list(zip(w, y_test, predictions))
                        w_zip.sort(key=lambda x: x[0])
                        w_zip.reverse()

                        w, y_test, predictions = zip(*w_zip)
                        w = np.array(w)
                        y_test = np.array(y_test)
                        predictions = np.array(predictions)
                        acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                        scores.append(acc)

                        roc_scores.append(roc_auc_score(y_test, w))


                if np.mean(roc_scores) > max_auroc[4]:
                    max_auroc = (C, gamma, np.mean(scores), np.std(scores), np.mean(roc_scores), np.std(roc_scores))

        # print("Max AUROC parameters:")
        # print("C =", max_auroc[0])
        # print("gamma =", max_auroc[1])
        # print("avg. accuracy: ", max_auroc[2])
        # print("std: ", max_auroc[3])
        # print("avg. AUROC: ", max_auroc[4])
        # print("std: ", max_auroc[5])

        max_auroc = list(max_auroc)
        table.loc[round_name] = max_auroc

    return table, data_info

    print("")


def normalize(K):

    K_sum = np.sum(K)  # get sum of K
    D = np.diag(np.ones(len(K)) * (1 / math.sqrt(K_sum)))  # construct D ^ -1/2
    K = D.dot(K).dot(D)  # compute normed K
    return K


def SVM_rbf(data_list):
    data, y = data_list[0], data_list[1]

    table = pd.DataFrame(
        columns=['C', 'gamma_img', 'gamma_wgt', 'avg. Accuracy', 'std.', 'avg. AUROC', 'std.'])  # for different gammas

    data_info = pd.DataFrame(columns=['# training samples', '# test samples', '# males', '# females', 'male/fem ratio'])

    print("# males:", len(np.where(y == 0)[0]))
    print("# females:", len(np.where(y == 1)[0]))
    print("SVM (separate RBF kernels) All features unnormalized")
    print("********************************************")
    print("")

    kf = StratifiedKFold(n_splits=10)

    X = data[0]

    X_img = X[:, :6]
    X_wgt = X[:, 6:]
    # test_img = X[:, :6]
    # test_wgt = X[:, 6:]

    print(len(X_img))
    print(len(X_wgt))


    scores = []
    z = 1

    gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]
    C_list = [0.001, 0.01, 0.1, 1, 10, 100]
    max_auroc = (0, 0, 0, 0, 0, 0, 0)
    N = len(X_img)

    gram_img = np.zeros((N,N))
    gram_wgt = np.zeros((N,N))

    for gamma_img in tqdm(gamma_list):
        for gamma_wgt in gamma_list:
            for C in C_list:

                # gram_img = expit(- distance_matrix(X_img, X_img)**2 * gamma_img)
                # gram_wgt = expit(- distance_matrix(X_wgt, X_wgt)**2 * gamma_wgt)

                # compute RBF kernel cell - by - cell (better way to do this?)
                for i in range(N):
                    for j in range(N):

                        gram_img[i][j] = np.exp(-gamma_img * np.linalg.norm(X_img[i] - X_img[j])**2)
                        gram_wgt[i][j] = np.exp(-gamma_wgt * np.linalg.norm(X_wgt[i] - X_wgt[j])**2)

                # gram_img = normalize(gram_img)
                # gram_wgt = normalize(gram_wgt)

                X_norm = 0.5 * gram_img + 0.5 * gram_wgt

                roc_scores = []

                for train_index, test_index in kf.split(X_norm, y):

                    X_train, X_test = X_norm[train_index], X_norm[test_index]
                    X_train = np.delete(X_train, test_index, axis=1)
                    X_test = np.delete(X_test, test_index, axis=1)



                    if z == 1:
                        data_list = [len(X_train), len(X_test), len(np.where(y == 0)[0]), len(np.where(y == 1)[0]),
                                     (len(np.where(y == 0)[0]) / len(y))]
                        z -= 1


                        data_info.loc['All rounds combined'] = data_list

                    y_train, y_test = y[train_index], y[test_index]

                    clf = make_pipeline(
                        SVC(C=C, kernel='precomputed',  class_weight='balanced', probability=True))

                    # train_img = X_train[:,:6]
                    # train_wgt = X_train[:,6:]
                    # test_img = X_test[:, :6]
                    # test_wgt = X_test[:, 6:]
                    #
                    # gram_train_img = expit(distance_matrix(train_img, train_img) * gamma_img)
                    # gram_train_wgt = expit(distance_matrix(train_wgt, train_wgt) * gamma_wgt)

                    # test_kernel_img = expit(distance_matrix(test_img, train_img) * gamma_img)
                    # test_kernel_wgt = expit(distance_matrix(test_wgt, train_wgt) * gamma_wgt)

                    # gram_train_img = normalize(gram_train_img)
                    # gram_train_wgt = normalize(gram_train_wgt)

                    # gram_train = gram_train_img + gram_train_wgt
                    # gram_test = test_kernel_img + test_kernel_wgt

                    clf.fit(X_train, y_train)

                    predictions = clf.predict(X_test)
                    acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                    scores.append(acc)

                    w = clf.decision_function(X_test)
                    w_zip = list(zip(w, y_test, predictions))
                    w_zip.sort(key=lambda x: x[0])
                    w_zip.reverse()
                    w, y_test, predictions = zip(*w_zip)
                    w = np.array(w)
                    y_test = np.array(y_test)
                    predictions = np.array(predictions)
                    acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                    scores.append(acc)

                    roc_scores.append(roc_auc_score(y_test, w))

                if np.mean(roc_scores) > max_auroc[5]:
                    max_auroc = (C, gamma_img, gamma_wgt, np.mean(scores), np.std(scores), np.mean(roc_scores), np.std(roc_scores))


    max_auroc = list(max_auroc)
    table.loc['All features combined'] = max_auroc

    return table, data_info


    print("")

def SVM_poly(data_list):

    data, y = data_list[0], data_list[1]
    table = pd.DataFrame(columns=['C', 'd=2', 'avg. Accuracy', 'std.', 'avg. AUROC', 'std.'])
    data_info = pd.DataFrame(columns=['# training samples', '# test samples', '# males', '# females', 'male/fem ratio'])

    print("# males:", len(np.where(y == 0)[0]))
    print("# females:", len(np.where(y == 1)[0]))
    print("SVM (polynomial kernels) All features")
    print("********************************************")
    print("")

    kf = StratifiedKFold(n_splits=10)

    X = data[0]

    X_img = X[:, :6]
    X_wgt = X[:, 6:]

    print(len(X_img))
    print(len(X_wgt))

    scores = []
    z = 1

    C_list = [0.001, 0.01, 0.1, 1, 10, 100]
    max_auroc = (0, 0, 0, 0, 0, 0)

    N = len(X_img)

    gram_img = np.zeros((N,N))
    gram_wgt = np.zeros((N,N))

    d = 2

    # compute polynomial kernel for both features
    for i in range(N):
        for j in range(N):
            gram_img[i][j] = (X_img[i].dot(X_img[j])) ** d
            gram_wgt[i][j] = (X_wgt[i].dot(X_wgt[j])) ** d

    gram_img = normalize(gram_img)
    gram_wgt = normalize(gram_wgt)

    X_norm = 0.5 * gram_img + 0.5 * gram_wgt


    for C in tqdm(C_list):

        roc_scores = []

        for train_index, test_index in kf.split(X_norm, y):

            X_train, X_test = X_norm[train_index], X_norm[test_index]
            X_train = np.delete(X_train, test_index, axis=1)
            X_test = np.delete(X_test, test_index, axis=1)

            if z == 1:
                print('here')
                data_list = [len(X_train), len(X_test), len(np.where(y == 0)[0]), len(np.where(y == 1)[0]),
                             (len(np.where(y == 0)[0]) / len(y))]
                z -= 1

                data_info.loc['All - Poly'] = data_list
                print(data_info)

            y_train, y_test = y[train_index], y[test_index]

            clf = make_pipeline(
                SVC(C=C, kernel='precomputed', class_weight='balanced', probability=True))

            clf.fit(X_train, y_train)

            predictions = clf.predict(X_test)
            acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
            scores.append(acc)

            w = clf.decision_function(X_test)
            w_zip = list(zip(w, y_test, predictions))
            w_zip.sort(key=lambda x: x[0])
            w_zip.reverse()
            w, y_test, predictions = zip(*w_zip)
            w = np.array(w)
            y_test = np.array(y_test)
            predictions = np.array(predictions)
            acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
            scores.append(acc)

            roc_scores.append(roc_auc_score(y_test, w))

        if np.mean(roc_scores) > max_auroc[5]:
            max_auroc = (C, d, np.mean(scores), np.std(scores), np.mean(roc_scores), np.std(roc_scores))


    max_auroc = list(max_auroc)
    table.loc['All features combined'] = max_auroc

    print("")

    return table, data_info





def main():
    rounds = ['all']
    for round in rounds:
        global ROUND
        ROUND = round
        filepath = '../data/measurements/round-' + str(round) + '.csv'
        data_list = getData(filepath)
        table, data_info = SVM_poly(data_list)
        print(data_info)
        print(table)
        table_name = '../results/tables/table-poly-N-' + str(round) + '.tex'
        with open(table_name, 'w') as tf:
            tf.write(data_info.to_latex().replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule','\\hline'))
            tf.write(table.to_latex().replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule','\\hline'))


if __name__ == '__main__':
    main()
