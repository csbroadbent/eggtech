import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import os
import plotly.graph_objects as go
from tqdm import tqdm

NO_LABELS = True
COMBINING = False

def combine_rounds(data_list, day):
    r3X, r3Y = data_list[0][0], data_list[0][1]
    r4X, r4Y = data_list[1][0], data_list[1][1]
    r5X, r5Y = data_list[2][0], data_list[2][1]

    X = np.concatenate((r3X, r4X, r5X), axis=0)
    y = np.concatenate((r3Y, r4Y, r5Y), axis=0)

    data = np.insert(X, 0, y, axis=1)

    savepath = "../data/voc/labeled/combined/diff/Days" + str(day) + "-combined.csv"
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

    if COMBINING == True:
        df = df.drop(['Data range'], axis=1)

        if df.columns[0] != 'label':
            df = df.drop(df.columns[0], axis=1)

    else:
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

        kf = StratifiedKFold(n_splits=5)

        for train_index, test_index in kf.split(X):  # inelegant way to print train and test size
            print("Training size:", len(train_index))
            print("Testing size:", len(test_index))
            break

        print("# males:", len(np.where(y == 0)[0]))
        print("# females:", len(np.where(y == 1)[0]))

        for k in k_list:

            scores = []

            for train_index, test_index in kf.split(X, y):

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

def SVM(folderpath, days='all',):
    for filename in os.listdir(folderpath):

        table = pd.DataFrame(
            columns=['C', 'gamma', 'avg. Accuracy', 'std.', 'avg. AUROC', 'std.'])  # for different gammas

        data_info = pd.DataFrame(
            columns=['# training samples', '# test samples', '# males', '# females', 'male/fem ratio'])

        if filename[-4:] != '.csv': #skip over folders
            continue


        if days != 'all':
            if filename[:-4] not in days:
                continue

        filepath = folderpath + '/' + filename
        X, y = get_data_matrix(filepath)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        print(filename[:-4])
        print("SVM")
        print("********************************************")
        print("")

        kf = StratifiedKFold(n_splits=10)

        for train_index, test_index in kf.split(X, y):  # inelegant way to print train and test size
            print("Training size:", len(train_index))
            print("Testing size:", len(test_index))
            break

        print("# males:", len(np.where(y == 0)[0]))
        print("# females:", len(np.where(y == 1)[0]))
        print("male/fem ratio:", (len(np.where(y == 0)[0]) / len(y)))

        print("")
        # print("LINEAR KERNEL")
        # print("______________________________________________________")

        scores = []
        #
        # for train_index, test_index in kf.split(X):
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        #
        #     clf = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
        #     clf.fit(X_train, y_train)
        #     predictions = clf.predict(X_test)
        #     acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
        #     scores.append(acc)
        #
        # print("Average accuracy is: ", np.mean(scores))
        # print("")
        # print("RBF KERNEL")
        # print("______________________________________________________")
        #
        # deg_list = [2,3,4,5,6,7]
        #

        # gamma_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.1, 0.5, 1]
        # C_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.1, 0.5, 1]

        j = 1


        gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]
        C_list = [0.001, 0.01, 0.1, 1, 10, 100]

        max_auroc = (0, 0, 0, 0, 0, 0)


        for gamma in tqdm(gamma_list):

            col = 0

            for C in C_list:

                scores = []
                roc_scores = []

                for train_index, test_index in kf.split(X, y):


                    y_train, y_test = y[train_index], y[test_index]

                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    if j == 1:

                        data_list = [len(X_train), len(X_test), len(np.where(y == 0)[0]), len(np.where(y == 1)[0]), (len(np.where(y == 0)[0]) / len(y))]
                        j -= 1

                        data_info.loc[filename] = data_list

                    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C =C ,gamma=gamma, class_weight='balanced', probability=True))
                    clf.fit(X_train, y_train)
                    predictions = clf.predict(X_test)

                    w = clf.decision_function(X_test)
                    w_zip = list(zip(w,y_test, predictions))
                    w_zip.sort(key =lambda x:x[0])
                    w_zip.reverse()
                    w, y_test, predictions = zip(*w_zip)
                    w = np.array(w)
                    y_test = np.array(y_test)
                    predictions = np.array(predictions)
                    acc = 1 - (len(y_test) - (y_test == predictions).sum()) / len(y_test)
                    scores.append(acc)

                    roc_scores.append(roc_auc_score(y_test, w))

                    # print("y_test")
                    # print(y_test)
                    # print("predictions")
                    # print(predictions)
                    # print("AUROC")
                    # print(roc_auc_score(y_test, w))
                    # print("accuracy")
                    # print(acc)
                    # print("")

                    # method I: plt
                    # y_train_pred = clf.decision_function(X_train)
                    # y_test_pred = clf.decision_function(X_test)
                    #
                    # train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
                    # test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
                    #
                    # plt.grid()
                    #
                    # plt.plot(train_fpr, train_tpr, label=" AUC TRAIN =" + str(auc(train_fpr, train_tpr)))
                    # plt.plot(test_fpr, test_tpr, label=" AUC TEST =" + str(auc(test_fpr, test_tpr)))
                    # plt.plot([0, 1], [0, 1], 'g--')
                    # plt.legend()
                    # plt.xlabel("True Positive Rate")
                    # plt.ylabel("False Positive Rate")
                    # plt.title("AUC(ROC curve)")
                    # plt.grid(color='black', linestyle='-', linewidth=0.5)
                    # plt.show()
                    # plt.close()


                if np.mean(roc_scores) > max_auroc[4]:
                    max_auroc = (C, gamma, np.mean(scores), np.std(scores), np.mean(roc_scores), np.std(roc_scores))

        max_auroc = list(max_auroc)
        table.loc[filename] = max_auroc

        table_name = '../results/tables/table-' + filename + '.tex'
        with open(table_name, 'w') as tf:
            tf.write(
                data_info.to_latex().replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace(
                    '\\bottomrule',
                    '\\hline'))
            tf.write(
                table.to_latex().replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule',
                                                                                                         '\\hline'))

        print("")



def combine():
    for i in range(1, 7):
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

def combine():
    r3filepath = '../data/voc/labeled/combined/diff/R3-Days1-2-diff.csv'
    r4filepath = '../data/voc/labeled/combined/diff/R4-Days1-2-diff.csv'
    r5filepath = '../data/voc/labeled/combined/diff/R5-Days1-2-diff.csv'

    r3 = [[], []]
    r4 = [[], []]
    r5 = [[], []]

    r3[0], r3[1] = get_data_matrix(r3filepath)

    r4[0], r4[1] = get_data_matrix(r4filepath)
    r5[0], r5[1] = get_data_matrix(r5filepath)
    data_list = [r3,r4,r5]
    combine_rounds(data_list, 12)

def plot():
    x = [1,2,3,4,5,6]
    y = [0.634, 0.705, 0.701, 0.671, 0.734, 0.634]
    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_xaxes(title_text='Day')
    fig.update_yaxes(title_text='Avg. AUROC')
    fig.show()

def main():
    folderpath = '../data/voc/labeled/combined'

    # plot()




    SVM(folderpath, 'all')


    # combine()





if __name__ == '__main__':
    main()