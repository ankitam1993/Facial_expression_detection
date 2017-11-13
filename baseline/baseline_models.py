import numpy as np
import csv
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

class baseline_models(object):

    def __init__(self):

        self.X_train = []
        self.Y_train = []

        self.X_val = []
        self.Y_val = []

        self.X_test = []
        self.Y_test = []

        self.model = None

    def load_data(self):

        X = []
        Y = []

        with open('../data/fer2013.csv', 'r') as f:
            data = csv.reader(f)

            for row in data:
                Y.append(row[0])
                z = list(row[1:-1][0].split())
                X.append(z)

        X = [a for a in X[1:]]
        Y = [a for a in Y[1:]]

        total_l = len(X)

        self.X_train = X[0:80*total_l/100]
        self.Y_train = Y[0:80*total_l/100]

        self.X_val = X[80*total_l/100:90*total_l/100]
        self.Y_val = Y[80*total_l/100:90*total_l/100]

        self.X_test = X[90*total_l/100:total_l]
        self.Y_test = Y[90*total_l/100:total_l]

        print 'length of x train and y train:' , len(self.X_train) , len(self.Y_train)
        print 'data loaded'

    def fit(self,model_name,kwargs):
        if model_name == 'svm':
            self.model = svm.SVC()

        elif model_name == 'rbf':
            self.model = RandomForestClassifier(**kwargs)

        print 'fitting the model...'
        self.model.fit(self.X_train, self.Y_train)

        print 'model fitting done...'

    def predict(self):

        print 'predicting the model'
        y_pred = self.model.predict(self.X_test)

        print 'Test set accuracy: ', (self.Y_test == y_pred).mean()






