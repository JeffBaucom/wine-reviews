from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

class WineClassifier(object):

    def __init__(self, weights, n_neighbors, categories):
        self.clf = RandomForestClassifier(n_estimators= n_neighbors)
        self.categories = categories
        print categories
        self.pca = PCA(n_components = 2) 
        self.scaler = preprocessing.StandardScaler()
        fig = plt.figure()
        self.ax1 = fig.add_subplot(111)

    def train(self, x_train, y_train):
        h = .02

        #x_train = np.asarray(x_train)

        new_x = np.zeros((len(x_train), len(x_train[0])))
        for i in range(0, len(x_train)):
            new_x[i] = np.reshape(x_train[i], len(x_train[0]))
        
        x_train = new_x
        self.scaler.fit(x_train)
        x_scaled = self.scaler.transform(x_train)
        X2D = self.pca.fit_transform(x_scaled)

        nY = []
        for i in y_train:
            tempList = i
            for j in range(len(tempList)):
                val = tempList[j]
                tempList[j] = self.categories.index(tempList[j])
            nY.append(sorted(tempList))

        X = np.array(X2D)
        y = preprocessing.MultiLabelBinarizer().fit_transform(nY)
        print cross_val_score(self.clf, X, y)
        self.ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
        return self.clf.fit(X, y)

    def predict(self, x_test):
        new_x = np.zeros((len(x_test), len(x_test[0])))
        for i in range(0, len(x_test)):
            new_x[i] = np.reshape(x_test[i], len(x_test[0]))
        
        x_test = new_x
        print x_test
        x_scaled = self.scaler.transform(x_test)
        X2D = self.pca.transform(x_scaled)

        self.ax1.scatter(X2D[:, 0], X2D[:, 1], c='r', alpha=0.5)
        return self.clf.predict(X2D)

    def show(self):
        plt.show()
