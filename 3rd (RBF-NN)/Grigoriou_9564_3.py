from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
from sklearn import metrics as mt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.decomposition import PCA
import pickle
import time
from pandas import DataFrame as df
import tensorflow as tf
from tensorflow.keras.models import Sequential


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
        from sklearn import svm
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
    # save_path = 'C:/Users/grist/Desktop/Neyrvnika/ex_3/'
    # model is the sklearn classification model object for evaluation
    # Xs ys are the validation or test data
    # model_name is the name of the model
    # Predicting the classes of the test Xs
    t = time.time()  # for calculating prediction time
    y_hat = modele.predict(Xs).astype(int)
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
    print('\n\n')
    # Calculating and plotting the confusion matrix of the model on the val/test data
    cm = mt.confusion_matrix(ys, y_hat)
    disp = mt.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names[0:len(label_names) - 1])
    disp.plot()
    plt.title(model_name)
    cm_name = model_name+'_cm.png'
    # plt.savefig(os.path.join(save_path,cm_name))
    plt.show()
    print('\n\n')
    # Classification examples exnum x len(label_names) grid of images with their
    # true label and their predicted label. This can be random or half correct
    # half wrong.
    # If no images where provided for presentation,classification example is
    # demanded and Xim=None (default) it is assumed that a PCA was performed.
    # (with sklearn.decomposition.PCA named pca). If Xim is an array-like variable
    # it is used for the examples. If it's a dictionary an error will occur.
    if hasattr(Xim, "__len__"):
        Xs = Xim
    elif Xim is None:
        Xs = pca.inverse_transform(Xs) * std + mu
        Xs = (Xs - np.min(Xs)) / (np.max(Xs) - np.min(Xs))
        Xs = np.reshape(Xs, (Xs.shape[0], 32, 32, 3))
    images = []
    y_pred = []
    y_true = []
    un_classes = np.unique(ys)
    exp_name = model_name+'_exp.png'
    if whoim == 'half':  # Half correct half wrong examples
        # Inserting a random factor for the images that will be shown as examples.
        random_factor = int(np.random.randint(0, high=int(np.floor(len(Xs) / \
                                                                   len(label_names) - 1)) * np.min(recall) - len(
            label_names) - exnum,
                                              size=(1,))[0])
        # Calculating the indices for each class.
        for j in range(len(label_names) - 1):
            ind_c = np.array(np.nonzero(np.logical_and(y_hat == ys, un_classes[j] == ys)))
            ind_c = np.reshape(np.transpose(ind_c)[0 + random_factor: \
                                                   random_factor + int(np.ceil(exnum / 2.0))],
                               (int(np.ceil(exnum / 2.0), )))
            ind_f = np.array(np.nonzero(np.logical_and(y_hat != ys, un_classes[j] == ys)))
            ind_f = np.reshape(np.transpose(ind_f)[0 + random_factor: \
                                                   random_factor + int(np.floor(exnum / 2.0))],
                               (int(np.floor(exnum / 2.0))))
            images.append(np.concatenate([Xs[ind_c], Xs[ind_f]]))
            y_pred.append(np.concatenate([y_hat[ind_c], y_hat[ind_f]]))
            y_true.append(np.concatenate([ys[ind_c], ys[ind_f]]))
            # Plotting a grid of examples
        figsize = (len(label_names) - 1, exnum + int(np.ceil(exnum / 2.0)))
        fig, axes = plt.subplots(exnum, len(label_names) - 1, figsize=figsize,tight_layout=True)
        fig.suptitle('Classification Examples ' + model_name)
        for i in range(axes.shape[1]):
            for j in range(axes.shape[0]):
                axes[j, i].imshow(images[i][j])
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])
                if j == 0:
                    if y_true[i][j] == -1:
                        axes[j, i].set_title(label_names[0])
                    else:
                        axes[j, i].set_title(label_names[y_true[i][j]])
                if y_true[i][j] == y_pred[i][j]:
                    if y_pred[i][j] == -1:
                        axes[j, i].set_xlabel(label_names[0], c='g')
                    else:
                        axes[j, i].set_xlabel(label_names[y_pred[i][j]], c='g')
                else:
                    if y_pred[i][j] == -1:
                        axes[j, i].set_xlabel(label_names[0], c='r')
                    else:
                        axes[j, i].set_xlabel(label_names[y_pred[i][j]], c='r')
        plt.show()
        print('\n\n')
    elif whoim == 'rand':  # Random examples
        random_factor = int(np.random.randint(0, high=int(np.floor(len(Xs) / \
                                                                   len(label_names) - 1)) - len(label_names) - exnum,
                                              size=(1,))[0])
        for j in range(len(label_names) - 1):
            ind = np.array(np.nonzero(un_classes[j] == ys))
            ind = np.reshape(np.transpose(ind)[random_factor: \
                                               (random_factor + exnum)], (exnum,))
            images.append(Xs[ind])
            y_pred.append(y_hat[ind])
            y_true.append(ys[ind])
        figsize = (5,5)#(len(label_names) - 1, exnum + int(np.ceil(exnum / 2.0)))
        fig, axes = plt.subplots(exnum, len(label_names) - 1, figsize=figsize, tight_layout=True)
        fig.suptitle('Classification Examples ' + model_name)
        for i in range(axes.shape[1]):
            for j in range(axes.shape[0]):
                axes[j, i].imshow(images[i][j])
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])
                if j == 0:
                    if y_true[i][j] == -1:
                        axes[j, i].set_title(label_names[0])
                    else:
                        axes[j, i].set_title(label_names[y_true[i][j]])
                if y_true[i][j] == y_pred[i][j]:
                    if y_pred[i][j] == -1:
                        axes[j, i].set_xlabel(label_names[0], c='g')
                    else:
                        axes[j, i].set_xlabel(label_names[y_pred[i][j]], c='g')
                else:
                    if y_pred[i][j] == -1:
                        axes[j, i].set_xlabel(label_names[0], c='r')
                    else:
                        axes[j, i].set_xlabel(label_names[y_pred[i][j]], c='r')

        # plt.savefig(os.path.join(save_path, exp_name))
        plt.show()
        # print('\n\n')
    return recall, precision, f1


def choose_k(features, kstart=2, kstop=20, kstep=1):
    # Function to visualize the moment and the silhouette score for different number of centers with k-means
    Krange = np.arange(start=kstart, stop=kstop, step=kstep)
    silhouettescores = []
    distortion = []
    for kval in Krange:
        means = KMeans(n_clusters=kval, random_state=73, n_init=1)
        print('k = %d starting...' % kval, end='')
        means.fit(features)
        print('fitted.\n')
        distortion.append(means.inertia_)
        silhouettescores.append(silhouette_score(features, means.labels_))

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(Krange, distortion, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(Krange, silhouettescores, marker='o')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()


class RBFLayer:
    ## Class of the RBF Hidden Layer with number of centers nodes and one output node
    def __init__(self, centers, normalized=False):
        # centers are the coordinates of the centers for the hidden nodes
        self.centers = centers
        # self.out_shape = (centers.shape[0]+1,) #if we want to add a bias phi
        self.out_shape = (centers.shape[0],)  # The output shape of the layer useful if we want just to train the hidden layer and output it to multiple output nodes
        self.weights = np.zeros(self.out_shape)  # The weights of each edge connecting the hidden nodes to the output node
        d_max = 0  # Calculating the maximum center distance in order to estimate the gamma parameter (Lowe,1989)
        for i in range(self.centers.shape[0]):
            for j in range(i + 1, self.centers.shape[0]):
                d_temp = np.linalg.norm(self.centers[i] - self.centers[j], 2)
                if d_temp > d_max:
                    d_max = d_temp
        # self.gamma = 1 / (2 * (d_max / np.sqrt((self.out_shape[0]-1) * 2)) ** 2) #if we want to add a bias phi
        self.gamma = 1 / (2 * (d_max / (np.sqrt((self.out_shape[0]) * 2))) ** 2)  # gamma = 1/(2*sigma^2)
        print('Gamma calculated.')
        self.class_weights = []  # class weights for unbalanced classes
        self.normalized = normalized

    def RLSstep(self, inputs):
        # This method unsupervisingly trains the hidden layer on the inputs given the self.centers
        # Its output is the result of the kernel functions on the inputs
        # phi = np.zeros((inputs.shape[0], self.out_shape[0]-1))#if we want to add a bias phi
        phi = np.zeros((inputs.shape[0], self.out_shape[0]))
        if self.normalized:
            for observation in range(inputs.shape[0]):
                phi[observation] = np.exp(
                    -self.gamma * np.sum((inputs[observation] - self.centers) ** 2, 1))  # Gaussian
                phi[observation] = phi[observation] / np.sum(phi[observation])
                print('%d out of %d' % (observation + 1, inputs.shape[0]), end='\r')
            print('\nPhis calculated.\n')
            # phi = np.hstack((phi,np.ones((inputs.shape[0],1))))#if we want to add a bias phi
        else:
            for observation in range(inputs.shape[0]):
                phi[observation] = np.exp(
                    -self.gamma * np.sum((inputs[observation] - self.centers) ** 2, 1))  # Gaussian
                print('\r%d out of %d' % (observation + 1, inputs.shape[0]), end='')
            print('\n')
            # phi = np.hstack((phi,np.ones((inputs.shape[0],1))))#if we want to add a bias phi
        return phi

    def fit(self, inputs, ys, xval=None, yval=None, epochs=2, regul=0.01, weighted="balanced", ready=None):
        # This method uses the RLS algorithm to train the weights of the hidden layer nodes
        # It works for binary classification with sign scheme (-1,1)
        # Its result is saved into the self.weights variable in order to be usable for later predictions
        # regul is the regularizer parameter lambda
        # The ready argument can be used to pass the phi values of theese inputs for the Layer and avoid multiple
        # unecesssary calculations
        # If weighted equals balanced then the inverse frequency weights are used during the training
        N = inputs.shape[0]
        Pn = (1 / regul) * np.eye(self.out_shape[0])  # The inverse of the correlation matrix
        if ready is None:
            phis = self.RLSstep(inputs)
        else:
            phis = ready
        if weighted == "balanced":
            self.class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(ys), y=ys)
        else:
            self.class_weights = np.ones((len(np.unique(ys)),))  # if not balanced we don't use weights
        for epoch in range(epochs):
            print('Epoch %d out of %d.' % (epoch + 1, epochs))
            for observation in range(N):
                print('\r%d/%d' % (observation + 1, N), end='')
                phi = phis[observation]
                Pn = Pn - (Pn @ np.outer(phi, phi) @ Pn) / (
                            1 + np.expand_dims(phi, 0) @ Pn @ phi)  # correlation matrix update
                gn = Pn @ phi  # gain vector
                if ys[observation] == -1:  # to add the correct weights BE CAREFUL to use -1,1 binary scheme
                    an = self.class_weights[0] * (ys[observation] - np.expand_dims(self.weights, 0) @ phi)
                else:
                    an = self.class_weights[1] * (ys[observation] - np.expand_dims(self.weights, 0) @ phi)
                # an = ys[observation] - np.expand_dims(self.weights,0)@phi
                self.weights = self.weights + gn * an  # weights update
            y_hat = (np.sum(self.weights * phis, 1))  # calculation of train metrics
            loss = (np.sum((ys - y_hat) ** 2) / 2 + regul * np.sum(
                self.weights ** 2) / 2) / N  # mean cost function is the squared error plus the regularization error
            # (MSE+sth)
            acc = 100 * np.sum(np.sign(y_hat) == ys) / len(ys)
            if xval is not None:  # We can print the validation set metrics ass well
                y_hat_val = self.predict(xval)
                val_loss = (np.sum((yval - y_hat_val) ** 2) / 2 + regul * np.sum(self.weights ** 2) / 2) / N
                val_acc = 100 * np.sum(y_hat_val == yval) / len(yval)
                print('train loss = %.5f\t train acc = %.2f\tvalidation loss = %.5f\tvalidation acc = %.2f' % (
                    loss, acc, val_loss, val_acc))
                print('\n')
            else:
                print('train loss = %.5f\t train acc = %.2f' % (loss, acc))
                print('\n')

    def predict(self, inputs, real=False, ready=None):
        # This method returns the predictions of the RBFLayer, they can be real numbers in order to be used in some multiclass scheme or
        # the binary classification result (the sign)
        if ready is None:
            phis = self.RLSstep(inputs)
        else:
            phis = ready
        if real:
            y_hat = (np.sum(self.weights * phis, 1))
        else:
            y_hat = np.sign(np.sum(self.weights * phis, 1))
        return y_hat


class RBFnn:
    # This class is used to either enable different method of the output layer weight training (using a 2 Layer Dense Neural Network)
    # or to enable 4 different multiclass schemes using the binary RBFLayer with RLS or NN weight training at the end.
    def __init__(self, classifier_code, k_centers, way='ovo', normalized=False):
        # self.mul_class = None
        self.rbfnodes = []  # The number of RBF output nodes of the Network (remeber RBFLayer by default has one "binary" output node)
        self.c_c = classifier_code  # This is an array with rows = number of classes used in OVR and ECC classification modes
        self.K = k_centers  # The number of centers for the RBF hidden layer
        self.train_out = None  # The output of the hidden layer stored to be used on the traning of the output layer
        self.train_labels = None  # The labels of the corresponding train_out
        self.predict_way = way  # The mode of the network(OVR,OVO,ECC,NN)
        self.normalized = normalized

    def fit(self, inputs, ys, xval=None, yval=None, weighted="balanced"):
        self.train_labels = ys
        if self.c_c is not None:
            hidden_layer = KMeans(n_clusters=self.K, random_state=42, n_init=1)
            centers = hidden_layer.fit(
                inputs)  # We use the same centers for all classifiers, this way it is like the hidden RBF Layer is connected to each
            # output node densely. (hidden nodes*out nodes = number of total weights)
            self.train_out = np.zeros((inputs.shape[0], self.c_c.shape[1]))
            first_node = RBFLayer(hidden_layer.cluster_centers_, normalized=self.normalized)
            phiss = first_node.RLSstep(inputs)  # Calculating the results of the hidden layer on the input
            for i in range(
                    self.c_c.shape[1]):  # The number of binary classifier on the scheme (or output nodes on the layer)
                node = self.c_c[:, i]
                y_temp = ys.copy()
                for j in range(len(node)):  # Binarizing the labels according to classifier scheme
                    if node[j] == 1:
                        y_temp[ys == j] = -1
                    else:
                        y_temp[ys == j] = 1
                rbf_node = RBFLayer(hidden_layer.cluster_centers_,
                                    normalized=self.normalized)  # Initializing the one node output
                print('Fitting %d out of %d' % (i + 1, self.c_c.shape[1]))
                rbf_node.fit(inputs, y_temp, xval, yval, 2, ready=phiss,
                             weighted=weighted)  # Calculating its weights using RLS
                self.train_out[:, i] = rbf_node.predict(inputs, True, ready=phiss)  # Saving its output
                self.rbfnodes.append(rbf_node)  # Saving the output_node/classifier
        else:  # If no classifier code is given we assume OVO
            classes = np.unique(ys)
            n = len(classes)
            self.train_out = np.zeros((inputs.shape[0], int((n * (
                        n - 1)) / 2)))  # In case of cifar-10 it is 45 output nodes/binary classifiers
            count = 0  # lazy way
            hidden_layer = KMeans(n_clusters=self.K, random_state=42, n_init=1)
            centers = hidden_layer.fit(
                inputs)  # We use the same centers for all classifiers, this way it is like the hidden RBF Layer is connected to each
            # output node densely. (hidden nodes*out nodes = number of total weights)
            first_node = RBFLayer(hidden_layer.cluster_centers_, normalized=self.normalized)
            phiss = first_node.RLSstep(inputs)  # Calculating the results of the hidden layer on the input
            for i in range(n):
                for j in range(i + 1,
                               n):  # For each pair of classes train a binary classifier/output node weights using RLS
                    y_temp = ys.copy()
                    x_temp = inputs.copy()
                    phis_temp = phiss.copy()
                    phis_temp = phis_temp[np.logical_or(y_temp == i, y_temp == j)]
                    x_temp = x_temp[np.logical_or(y_temp == i, y_temp == j)]
                    y_temp = y_temp[np.logical_or(y_temp == i, y_temp == j)]
                    y_temp[y_temp == i] = -1
                    y_temp[y_temp == j] = 1
                    rbf_node = RBFLayer(hidden_layer.cluster_centers_, normalized=self.normalized)
                    print('Fitting %d out of %d' % (count + 1, int((n * (n - 1)) / 2)))
                    rbf_node.fit(x_temp, y_temp, xval, yval, 2, ready=phis_temp, weighted=weighted)
                    self.train_out[:, count] = rbf_node.predict(inputs, True, ready=phiss)  # Save its output
                    self.rbfnodes.append(rbf_node)
                    count = count + 1  # Counting the nodes

    def predict(self, inputs):
        # This method is used in order to take a decision using a multiclass scheme.
        y_hat_t = np.zeros((inputs.shape[0], len(self.rbfnodes)))
        phiss = self.rbfnodes[0].RLSstep(inputs)  # Calculating the output of the hidden layer on the input
        for i in range(len(self.rbfnodes)):
            y_hat_t[:, i] = self.rbfnodes[i].predict(inputs, real=True,
                                                     ready=phiss)  # Calculating the output of each node on the input
        if self.predict_way == "ecc":  # Using error correcting classification scheme with soft decision (min Euclidean distance)
            dmin = 1e10
            y_hat = np.zeros((inputs.shape[0], self.c_c.shape[0]))
            for i in range(self.c_c.shape[0]):  # Calculating the distance from each class
                y_hat[:, i] = np.sum((y_hat_t - self.c_c[i]) ** 2, 1)
            y_hat = np.argmin(y_hat, axis=1)  # Choosing class
        elif self.predict_way == 'ovo':  # Using one vs one strategy
            minus1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                      4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7,
                      8]  # The way they are stored (only works on 10 label class at the moment)
            plus1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 5, 6, 7,
                     8, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9]
            y_hat_t = np.sign(y_hat_t)
            for i in range(len(self.rbfnodes)):  # Hard voting
                y_hat_t[y_hat_t[:, i] == 1, i] = plus1[i]
                y_hat_t[y_hat_t[:, i] == -1, i] = minus1[i]
            y_hat = np.zeros((inputs.shape[0],))
            for i in range(inputs.shape[0]):
                y_hat[i] = np.argmax(np.bincount(np.round(y_hat_t[i]).astype(int)))  # Voting
        elif self.predict_way == 'ovr':  # One vs rest scheme (classification code is a np.eye(len(np.unique(labels))) matrix)
            y_hat_t = (y_hat_t - np.mean(y_hat_t, 0)) / np.std(y_hat_t, 0)
            y_hat = np.argmin(y_hat_t, 1)  # Voting (minimum because of how the output nodes are trained)
        elif self.predict_way == 'NN':  # Else we train a one layer softmax network to decide on the real outputs of the RBF layer
            pred_model = Sequential()
            pred_model.add(tf.keras.layers.InputLayer(self.train_out.shape[1], ))
            pred_model.add(tf.keras.layers.Dense(500, kernel_initializer='glorot_normal', activation='tanh'))
            pred_model.add(tf.keras.layers.Dense(10, activation='softmax'))
            pred_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='categorical_crossentropy',
                               metrics=['categorical_accuracy'])
            hist = pred_model.fit(self.train_out, tf.one_hot(self.train_labels, 10), 50, 50, verbose=1)
            y_hat = np.argmax(pred_model.predict(y_hat_t), 1)
        else:
            y_hat = y_hat_t
        return y_hat


class binary_model:
    #Class to automate and store the Kmeans,RLS trained RBF network for binary classification
    def __init__(self, num_of_centers, way='kmeans', normalized=True, lambda_par=0.25):
        self.center_way = way
        self.centers = []
        self.num_of_centers = num_of_centers  # number of rb functions
        self.normalized = normalized
        self.lambda_par = lambda_par
        self.rbf = []

    def fit(self, inputs, ys, xval=None, yval=None):
        if self.center_way == 'kmeans':  # we use kmeans to determine the centers of the rbfs
            kmt = KMeans(n_clusters=self.num_of_centers, random_state=73, n_init=1)
            kmt.fit(inputs)
            self.centers = kmt.cluster_centers_
        else:  # Use random samples as centers (equal from each class)
            minus_S = inputs[ys == -1].copy()
            plus_S = inputs[ys == 1].copy()
            minus_S = np.random.permutation(minus_S)
            plus_S = np.random.permutation(plus_S)
            minus_S = minus_S[0:int(np.floor(self.num_of_centers / 2))]
            plus_S = plus_S[0:int(np.ceil(self.num_of_centers / 2))]
            self.centers = np.vstack((minus_S, plus_S))
        print('Centers Calculated')
        self.rbf = RBFLayer(self.centers, normalized=self.normalized)
        self.rbf.fit(inputs, ys, xval, yval, regul=self.lambda_par)

    def predict(self, inputs):
        return self.rbf.predict(inputs)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

my_path = ''  # Change this to the path where your cifar-10 data is stored or keep it '' in order to download them.

if my_path == '':
    from keras.datasets import cifar10

    (X, y), (X_test, y_test) = cifar10.load_data()
    label = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    X = np.reshape(X, (50000, 3072))
    X_test = np.reshape(X_test, (10000, 3072))
    y = np.reshape(y, (50000,))
    y_test = np.reshape(y_test, (10000,))
else:
    X, y, X_test, y_test, label = load_data(my_path)
    label[0] = 'plane'  # for visual purposes
    label[1] = 'auto'
# Z-score scaling of the data for PCA implementation
mu = np.mean(X[np.logical_or(y == 1, y == 9)], axis=0)
std = np.std(X[np.logical_or(y == 1, y == 9)], axis=0)
X_pca = (X[np.logical_or(y == 1, y == 9)] - mu) / std
X_test_pca = (X_test[np.logical_or(y_test == 1, y_test == 9)] - mu) / std
# PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X_pca)
X_test_pca = pca.transform(X_test_pca)
if my_path != '':
    X_test = np.transpose(np.reshape(X_test, (10000, 3, 32, 32)), (0, 2, 3, 1))  # for image plotting


# Xval = X_test_pca.copy()
# Yval = y_test.copy()
# y_ap = y.copy()
# X_ap = X_pca.copy()
X_dh = X_pca.copy()
y_dh = y.copy()
y_dh = y_dh[np.logical_or(y_dh == 1, y_dh == 9)]
y_dh = y_dh
X_dht = X_test_pca.copy()
y_dht = y_test.copy()
y_dht = y_dht[np.logical_or(y_dht == 1, y_dht == 9)]
y_dh[y_dh == 1] = -1
y_dh[y_dh == 9] = 1
y_dht[y_dht == 1] = -1
y_dht[y_dht == 9] = 1

# params = {'k':[100,250,500,1000],'way':['kmeans','random_samples']}#
# for k in params['k']:
#     for way in params['way']:
#         name = 'MPLA_%s_%d_neurons'% (way,k)
#         bm = binary_model(num_of_centers=k,way=way)
#         t_1 = time.time()  # for runtime estimation
#         bm.fit(X_dh, y_dh)
#         t_1 = time.time() - t_1
#         print('%s:' % name)
#         print('Training time elapsed: %.2f ms.' % (1000 * t_1))
#         print(name+' train')
#         _, _, _ = evaluate_model(bm, X_dh, y_dh, name+' train', ['auto', 'truck'],
#                                  X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)#
#         print(name+' test')
#         _, _, _ = evaluate_model(bm, X_dht, y_dht, name + ' test', ['auto', 'truck'],
#                                  X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)
kc = 235
name = '%d_neurons_kmeans'%kc
bm = binary_model(num_of_centers=kc,way='kmeans')
t_1 = time.time()  # for runtime estimation
bm.fit(X_dh, y_dh)
t_1 = time.time() - t_1
print('%s:' % name)
print('Training time elapsed: %.2f ms.' % (1000 * t_1))
print(name+' train')
_, _, _ = evaluate_model(bm, X_dh, y_dh, name+' train', ['auto', 'truck'],
                         X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)#
print(name+' test')
_, _, _ = evaluate_model(bm, X_dht, y_dht, name + ' test', ['auto', 'truck'],
                         X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)

##'Benchmark' models
nn1,nn1_name = train_model(X_dh,y_dh,'NN',kN=1)
#Evaluating 1-NN on test set
_, _, _ = evaluate_model(nn1, X_dht, y_dht, nn1_name + ' test', ['auto', 'truck'],
                         X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)
_,_,_ = evaluate_model(nn1,X_dh,y_dh,nn1_name+'train',['auto','truck'],
                       X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)
nn3,nn3_name = train_model(X_dh,y_dh,'NN',kN=3)
#Evaluating 1-NN on test set
_, _, _ = evaluate_model(nn3, X_dht, y_dht, nn3_name + ' test', ['auto', 'truck'],
                         X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)
_,_,_ = evaluate_model(nn3,X_dh,y_dh,nn3_name+'train',['auto','truck'],
                       X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)

nc,nc_name = train_model(X_dh,y_dh,'nearest_centroid',kN=3)
#Evaluating 1-NN on test set
_, _, _ = evaluate_model(nc, X_dht, y_dht, nc_name + ' test', ['auto', 'truck'],
                         X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)
_,_,_ = evaluate_model(nc,X_dh,y_dh,nc_name+'train',['auto','truck'],
                       X_test[np.logical_or(y_test == 1, y_test == 9)], whoim='rand',exnum=2)