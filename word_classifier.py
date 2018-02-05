from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class WordClassifier(object):

    def __init__(self, weights, n_neighbors, categories):
        self.clf = RandomForestClassifier(n_estimators=n_neighbors)
        self.categories = categories
        print categories
        self.pca = PCA(n_components = 3) 
        self.scaler = preprocessing.StandardScaler()

    def train(self, x_train, y_train):
        h = .02

        #x_scaled = preprocessing.scale(x_train)
        self.scaler.fit(x_train)
        x_scaled = self.scaler.transform(x_train)
        X2D = self.pca.fit_transform(x_scaled)



        nY = []
        for i in y_train:
            tempList = list(i[1])
            for j in range(len(tempList)):
                val = tempList[j]
                tempList[j] = self.categories.index(tempList[j])
            nY.append(sorted(tempList))

        X = np.array(X2D)
        y = preprocessing.MultiLabelBinarizer().fit_transform(nY)
        print cross_val_score(self.clf, X, y)
        return self.clf.fit(X, y)

    def test(self, x_test, y_test):
        x_scaled = self.scaler.transform(x_test)
        X2D = self.pca.transform(x_scaled)
        print y_test
        y_predict = self.clf.predict(X2D)
        print y_test
        print y_predict
        #print accuracy_score(y_test, y_predict)

    def predict(self, x_test):
        X2D = self.pca.transform(x_test)
        return self.clf.predict(X2D)

