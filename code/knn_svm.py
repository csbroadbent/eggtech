import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
ROUND = -1

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
    labels_training = pd.factorize(data_training['label'])[0]
    print("males: ", len(data_training[data_training['label'] == 'M']))
    print("females: ", len(data_training[data_training['label'] == 'F']))
    # format data into numpy arrays
    if ROUND == 3:
        data_training_array = np.column_stack((data_training['length'], data_training['width'],
                                               data_training['L-length'], data_training['T-length'],
                                               data_training['L-width'],
                                               data_training['T-width'], data_training['weight-day1'],
                                               data_training['weight-day2'], data_training['weight-day3'],
                                               data_training['weight-day4'],data_training['weight-day5'],
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

    print("# males:", len(np.where(y == 0)[0]))
    print("# females:", len(np.where(y == 1)[0]))

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
        kf = StratifiedKFold(n_splits=10)

        X = data[i]

        print("**********************************************")
        print("")
        print("LINEAR KERNEL")
        print("______________________________________________________")

        scores = []
        j=1

        # for train_index, test_index in kf.split(X):
        #
        #     X_train, X_test = X[train_index], X[test_index]
        #
        #     if j == 1:
        #         print("# training samples:", len(X_train))
        #         print("# test samples:", len(X_test))
        #         j -= 1
        #
        #     y_train, y_test = y[train_index], y[test_index]
        #
        #     if i < 4:
        #         clf = SVC(kernel='linear', class_weight='balanced')
        #         clf.fit(X_train, y_train)
        #         predictions = clf.predict(X_test)
        #         acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
        #         scores.append(acc)
        #
        #     else:
        #         clf = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
        #         clf.fit(X_train, y_train)
        #         predictions = clf.predict(X_test)
        #         acc = 1 - (len(y_test) - (y_test == predictions).sum()) /len(y_test)
        #         scores.append(acc)

        print("Average rate is: ", np.mean(scores))
        print("")
        print("RBF KERNEL")
        print("______________________________________________________")

        gamma_list = [0.001,0.01,0.1,1,10,100, 1000, 10000, 100000]
        j = 1
        for gamma in gamma_list:

            scores = []


            roc_avg = 0

            prec_avg = 0
            rec_avg = 0
            fscore_avg = 0
            sup_avg = 0

            for train_index, test_index in kf.split(X, y):

                X_train, X_test = X[train_index], X[test_index]

                if j == 1:
                    print("# training samples:", len(X_train))
                    print("# test samples:", len(X_test))
                    j -= 1

                y_train, y_test = y[train_index], y[test_index]

                # if i < 4:
                #     clf = SVC(kernel='rbf', gamma=gamma, class_weight='balanced')
                #     clf.fit(X_train, y_train)
                #     predictions = clf.predict(X_test)
                #     acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                #     scores.append(acc)

                if i > -1:
                    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=100, gamma=gamma, class_weight={1:2}, probability=True))
                    clf.fit(X_train, y_train)
                    predictions = clf.predict(X_test)
                    acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                    scores.append(acc)

                    # cf = pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predicted Male', "Predicted Female"],
                    #              index=['Actual Male', 'Actual Female'])

                    # print(cf)

                    # tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                    # probas = clf.predict_proba(X_test)[:, 1]
                    #
                    # def get_preds(threshold, probabilities):
                    #     return [1 if prob > threshold else 0 for prob in probabilities]

                    # roc_values = []
                    # for thresh in np.linspace(0, 1, 100):
                    #     predictions = get_preds(thresh, probas)
                    #     tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                    #     tpr = tp / (tp + fn)
                    #     fpr = fp / (fp + tn)
                    #     roc_values.append([tpr, fpr])
                    # tpr_values, fpr_values = zip(*roc_values)

                    # fig, ax = plt.subplots(figsize=(10, 7))
                    # ax.plot(fpr_values, tpr_values)
                    # ax.plot(np.linspace(0, 1, 100),
                    #         np.linspace(0, 1, 100),
                    #         label='baseline',
                    #         linestyle='--')
                    # plt.title('Receiver Operating Characteristic Curve', fontsize=18)
                    # plt.ylabel('TPR', fontsize=16)
                    # plt.xlabel('FPR', fontsize=16)
                    # plt.legend(fontsize=12);
                    # plt.close()

                    roc_avg += roc_auc_score(y_test, predictions)
                    # prec, rec, fscore, sup = precision_recall_fscore_support(y_test, predictions)
                    # prec_avg += prec
                    # rec_avg += rec
                    # fscore_avg += fscore
                    # sup_avg += sup

            print("gamma=", gamma," | Average accuracy:", np.mean(scores))
            print("Avg ROC:", roc_avg / 10)
            # print("Avg prec:", prec_avg / 10)
            # print("Avg rec:", rec_avg / 10)
            # print("Avg fscore:", fscore_avg / 10)
            # print("Avg sup:", sup_avg / 10)
        print("")


def main():
    data_list = getData('../data/measurements/rounds-combined.csv')
    # KNN(data_list)
    SVM(data_list)
    # LR(data_list)

if __name__ == '__main__':
    main()
