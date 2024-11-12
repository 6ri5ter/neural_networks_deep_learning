#           Grigoriou Stergios 9564                   grigster@ece.auth.gr
# Neural Networks - Deep Learning @csd.auth            2nd Project (SVMs)
#
# This .py file is essentially the code for the project. (The hyperparameter tuning of the
# SVMs is shown on the .ipynb file)

import os
import numpy as np
from sklearn import metrics as mt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import svm
import pickle
import time
from pandas import DataFrame as df

def unpickle(file):
    # This function turns pickle into dict, file is the file's path.
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='latin1')
    # batch is the dict containing the corresponding batch's data
    return batch


def load_data(path):
    # path is the path of the folder that contains cifar-10 uncompressed data files suitable for python
    # data contains the image data of the batch
    data = []
    # labels contains the corresponding to data labels
    labels = []
    metadata = unpickle(os.path.join(path, 'batches.meta'))
    label_names = metadata['label_names']
    m = metadata['num_vis']
    n = metadata['num_cases_per_batch']
    # loading all 5 batches
    for i in range(1, 6):
        filepath = os.path.join(path, 'data_batch_%d' % i)
        temp = unpickle(filepath)
        data.append(np.reshape(temp['data'], (n, m)))
        labels.append(np.array(temp['labels']))
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    # loading the test data
    temp = unpickle(os.path.join(path, 'test_batch'))
    test_data = np.reshape(temp['data'], (n, m))
    test_labels = np.array(temp['labels'])
    return data, labels, test_data, test_labels, label_names


def train_model(Xs, ys, model_type, svm_param=None, kN=0):
    # Xs and ys are the train set
    # model_type can be either 'NN' or 'nearest_centroid'
    # k is the k parameter (number of neighbours) for the k-NN model
    # svm_param is a dictionary with the necessary parameters of the model
    model_name = ''
    modele = KNeighborsClassifier(n_neighbors=kN, algorithm='brute')
    t = time.time()  # for runtime estimation
    # fitting the model to the train data
    if model_type == 'NN':
        modele = KNeighborsClassifier(n_neighbors=kN, algorithm='brute')
        modele.fit(Xs, ys)
        t = time.time() - t
        model_name = ('%d-NN' % kN)
    elif model_type == 'nearest_centroid':
        modele = NearestCentroid()
        modele.fit(Xs, ys)
        t = time.time() - t
        model_name = 'Nearest Centroid'
    elif model_type == 'svm':
        if svm_param['kernel'] == 'linear':
            modele = svm.LinearSVC(C=svm_param['C'], dual='auto')
            model_name = 'LSVM (C: %.2e)' % svm_param['C']
        elif svm_param['kernel'] == 'rbf':
            modele = svm.SVC(C=svm_param['C'], kernel='rbf', gamma=svm_param['gamma'])
            model_name = 'SVM (RBF, C: %.2e, gamma: %.2e)' % (svm_param['C'], svm_param['gamma'])
        elif svm_param['kernel'] == 'poly':
            modele = svm.SVC(C=svm_param['C'], degree=svm_param['degree'], gamma=svm_param['gamma'], kernel='poly')
            model_name = 'SVM (polynomial kernel of degree %d (C: %.2e, gamma: %.2e)' % (
            svm_param['degree'], svm_param['C'], svm_param['gamma'])
        elif svm_param['kernel'] == 'sigmoid':
            modele = svm.SVC(C=svm_param['C'], gamma=svm_param['gamma'], kernel='sigmoid')
            model_name = 'SVM (sigmoid, C: %.2e, gamma: %.2e)' % (svm_param['C'], svm_param['gamma'])
        modele.fit(Xs, ys)
        t = time.time() - t
    else:
        print('Wrong model_type input. Acceptable values\nare: "NN" or "nearest_centroid".')
    print('%s:' % model_name)
    print('Training time elapsed: %.2f ms.' % (1000 * t))
    return modele, model_name


def evaluate_model(modele, Xs, ys, model_name, label_names, Xim=None, exnum=5, whoim=None):
    # model is the sklearn classification model object for evaluation
    # Xs ys are the validation or test data
    # model_name is the name of the model
    # Predicting the classes of the test Xs
    t = time.time()  # for calculating prediction time
    y_hat = modele.predict(Xs)
    if len(y_hat.shape) > 1:
        y_hat = np.argmax(y_hat, 1)
    t = time.time() - t
    print('Prediction time elapsed: %.2f ms.' % (1000 * t))
    # Calculating recall and precision class wise
    recall = mt.recall_score(ys, y_hat, average=None)
    precision = mt.precision_score(ys, y_hat, average=None)
    # Calculating class wise f1_score
    f1 = 2 * recall * precision / (recall + precision)
    # Calculating average (macro) accuracy of the model
    accu = np.sum(np.equal(ys, y_hat)) / len(ys)  # or just the np.mean(recall)
    print('Accuracy: %.4f' % accu)
    daf = np.vstack((recall, precision))
    daf = np.vstack((daf, f1))
    avrgs = np.zeros((3, 1))
    avrgs[2] = np.mean(f1)
    avrgs[1] = np.mean(precision)
    avrgs[0] = np.mean(recall)
    daf = np.hstack((daf, avrgs))
    label_names = np.hstack((label_names, 'average'))
    daaf = df(data=daf, index=['recall', 'precision', 'f1'], columns=label_names)
    print(daaf.to_string())
    # print('\n\n')
    # Calculating and plotting the confusion matrix of the model on the val/test data
    cm = mt.confusion_matrix(ys, y_hat)
    disp = mt.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names[0:10])
    disp.plot()
    plt.title(model_name)
    plt.show()
    print('\n\n')
    # Classification examples exnum x len(label_names) grid of images with their
    # true label and their predicted label. This can be random or half correct
    # half wrong.
    # If no images where provided for presentation,classification example is
    # demanded and Xim=None (default) it is assumed that a PCA was performed.
    # (with sklearn.decomposition.PCA named pca). If Xim is an array-like variable
    # it is used for the examples. If its a dictionary an error will occur.
    if hasattr(Xim, "__len__"):
        Xs = Xim
    elif Xim is None:
        Xs = pca.inverse_transform(Xs) * std + mu
        Xs = (Xs - np.min(Xs)) / (np.max(Xs) - np.min(Xs))
        Xs = np.reshape(Xs, (Xs.shape[0], 32, 32, 3))
    images = []
    y_pred = []
    y_true = []
    if whoim == 'half':  # Half correct half wrong examples
        # Inserting a random factor for the images that will be shown as examples.
        random_factor = int(np.random.randint(0, high=int(np.floor(len(Xs) / \
                                                                   len(label_names) - 1)) * np.min(recall) - 10 - exnum,
                                              size=(1,))[0])
        # Calculating the indices for each class.
        for j in range(len(label_names) - 1):
            ind_c = np.array(np.nonzero(np.logical_and(y_hat == ys, j == ys)))
            ind_c = np.reshape(np.transpose(ind_c)[0 + random_factor: \
                                                   random_factor + int(np.ceil(exnum / 2.0))],
                               (int(np.ceil(exnum / 2.0), )))
            ind_f = np.array(np.nonzero(np.logical_and(y_hat != ys, j == ys)))
            ind_f = np.reshape(np.transpose(ind_f)[0 + random_factor: \
                                                   random_factor + int(np.floor(exnum / 2.0))],
                               (int(np.floor(exnum / 2.0))))
            images.append(np.concatenate([Xs[ind_c], Xs[ind_f]]))
            y_pred.append(np.concatenate([y_hat[ind_c], y_hat[ind_f]]))
            y_true.append(np.concatenate([ys[ind_c], ys[ind_f]]))
            # Plotting a the grid of examples
        figsize = (len(label_names) - 1, exnum + int(np.ceil(exnum / 2.0)))
        fig, axes = plt.subplots(exnum, len(label_names) - 1, figsize=figsize, tight_layout=True)
        fig.suptitle('Classification Examples ' + model_name)
        for i in range(axes.shape[1]):
            for j in range(axes.shape[0]):
                axes[j, i].imshow(images[i][j])
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])
                if j == 0:
                    axes[j, i].set_title(label_names[y_true[i][j]])
                if y_true[i][j] == y_pred[i][j]:
                    axes[j, i].set_xlabel(label_names[y_pred[i][j]], c='g')
                else:
                    axes[j, i].set_xlabel(label_names[y_pred[i][j]], c='r')
        plt.show()
        print('\n\n')
    elif whoim == 'rand':  # Random examples
        random_factor = int(np.random.randint(0, high=int(np.floor(len(Xs) / \
                                                                   len(label_names) - 1)) - 10 - exnum, size=(1,))[0])
        for j in range(len(label_names) - 1):
            ind = np.array(np.nonzero(j == ys))
            ind = np.reshape(np.transpose(ind)[random_factor: \
                                               (random_factor + exnum)], (exnum,))
            images.append(Xs[ind])
            y_pred.append(y_hat[ind])
            y_true.append(ys[ind])
        figsize = (len(label_names) - 1, exnum + int(np.ceil(exnum / 2.0)))
        fig, axes = plt.subplots(exnum, len(label_names) - 1, figsize=figsize, tight_layout=True)
        fig.suptitle('Classification Examples ' + model_name)
        for i in range(axes.shape[1]):
            for j in range(axes.shape[0]):
                axes[j, i].imshow(images[i][j])
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])
                if j == 0:
                    axes[j, i].set_title(label_names[y_true[i][j]])
                if y_true[i][j] == y_pred[i][j]:
                    axes[j, i].set_xlabel(label_names[y_pred[i][j]], c='g')
                else:
                    axes[j, i].set_xlabel(label_names[y_pred[i][j]], c='r')
        plt.show()
        print('\n\n')
    return recall, precision, f1


def gridsearch(Xs, ys, cv=3, kernel='rbf', C=2.0 ** np.arange(start=-5, stop=17, step=2), gamma=None, degree=None,
               p=True):
    # Function for hyperparameter tuning through CV
    grid_params = {}
    if hasattr(kernel, '__len__') and (not isinstance(kernel, str)):
        grid_params.update({'kernel': kernel})
    else:
        grid_params.update({'kernel': [kernel]})
    if hasattr(C, '__len__'):
        grid_params.update({'C': C})
    else:
        grid_params.update({'C': [C]})
    if hasattr(gamma, '__len__') and gamma is not None:
        grid_params.update({'gamma': gamma})
    elif gamma is not None:
        grid_params.update({'gamma': [gamma]})
    if hasattr(degree, '__len__') and degree is not None:
        grid_params.update({'degree': degree})
    elif degree is not None:
        grid_params.update({'degree': [degree]})
    if kernel == 'linear':
        del grid_params['kernel']
        cvSVM = GridSearchCV(svm.LinearSVC(dual='auto'), grid_params, scoring='accuracy', verbose=1, cv=cv, n_jobs=-1)
    else:
        cvSVM = GridSearchCV(svm.SVC(), grid_params, scoring='accuracy', verbose=1, cv=cv, n_jobs=-1)
    cvSVM.fit(Xs, ys)
    if p:
        bp = cvSVM.best_params_
        print(bp)
    return cvSVM


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

my_path = 'C:\cifar10'  # Change this to the path where your cifar-10 data is stored or to '' in order to download them.
if my_path == '':
    from keras.datasets import cifar10
    (X,y),(X_test,y_test) = cifar10.load_data()
    label = ['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']
    X = np.reshape(X,(50000,3072))
    X_test = np.reshape(X_test,(10000,3072))
    y = np.reshape(y,(50000,))
    y_test= np.reshape(y_test,(10000,))
else:
    X,y,X_test,y_test,label = load_data(my_path)
    label[0] = 'plane'#for visual purposes
    label[1] = 'auto'
#Z-score scaling of the data for PCA implementation
mu = np.mean(X,axis=0)
std = np.std(X,axis=0)
X_pca = (X-mu)/std
X_test_pca = (X_test-mu)/std
#PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X_pca)
X_test_pca = pca.transform(X_test_pca)
if my_path != '':
    X_test = np.transpose(np.reshape(X_test,(10000,3,32,32)),(0,2,3,1))#for image plotting
# Training nearest_centroid on train set
cn, cn_name = train_model(X_pca, y, 'nearest_centroid')
# Evaluating nearest_centroid on test set
_, _, _ = evaluate_model(cn, X_test_pca, y_test, cn_name, label, X_test, whoim='rand')
# Training 1-NN on train set
nn1, nn1_name = train_model(X_pca, y, 'NN', kN=1)
# Evaluating 1-NN on test set
_, _, _ = evaluate_model(nn1, X_test_pca, y_test, nn1_name, label, X_test, whoim='rand')
# Training 3-NN on train set
nn3, nn3_name = train_model(X_pca, y, 'NN', kN=3)
# Evaluating 3-NN on test set
_, _, _ = evaluate_model(nn3, X_test_pca, y_test, nn3_name, label, X_test, whoim='rand')
# Training tuned Linear SVM on train set
linSVM, lin_name = train_model(X_pca, y, 'svm', {'kernel': 'linear', 'C': 0.00011421052631578947})
# Evaluating Linear SVM on test set
_, _, _ = evaluate_model(linSVM, X_test_pca, y_test, lin_name, label, X_test, whoim='rand')
# Training tuned RBF SVM on train set
rbfSVM, rbf_name = train_model(X_pca, y, 'svm',
                               {'C': 3.363585661014858, 'gamma': 0.0006905339660024879, 'kernel': 'rbf'})
# Evaluating RBF SVM on test set
_, _, _ = evaluate_model(rbfSVM, X_test_pca, y_test, rbf_name, label, X_test, whoim='rand')
# Training tuned polynomial SVM on train set
polySVM, poly_name = train_model(X_pca, y, 'svm', {'C': 0.03125, 'degree': 3, 'gamma': 0.001953125,
                                                   'kernel': 'poly'})
# Evaluating polynomial SVM on test set
_, _, _ = evaluate_model(polySVM, X_test_pca, y_test, poly_name, label, X_test, whoim='rand')
# Training tuned sigmoid SVM on train set
sigSVM, sig_name = train_model(X_pca, y, 'svm',
                               {'C': 39.39662122703734, 'gamma': 1.0789593218788873e-05, 'kernel': 'sigmoid'})
# Evaluating sigmoid SVM on test set
_, _, _ = evaluate_model(sigSVM, X_test_pca, y_test, sig_name, label, X_test, whoim='rand')
