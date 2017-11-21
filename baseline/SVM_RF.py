import numpy as np
import csv
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from extract_features import *

# This class is implementing SVM , Random Forest with normal pixel data and with HOG +
# Color histogram features ( taken from assignment 1)

class baseline_models(object):

    def __init__(self):

        self.X_train = []
        self.Y_train = []

        self.X_val = []
        self.Y_val = []

        self.X_test = []
        self.Y_test = []

        self.model = None

    def load_data(self,bool):

        X = []
        Y = []

        print 'loading data'

        with open('../data/fer2013.csv', 'r') as f:
            data = csv.reader(f)

            for row in data:

                if row[0] != 'emotion':
                    Y.append(int(row[0]))
                    z = list(row[1:-1][0].split())
                    z = [int(a) for a in z]
                    X.append(z)
                else:
                    print row

        total_l = len(X)

        self.X_train = np.asarray(X[0:80 * total_l / 100], dtype=np.float32)
        self.Y_train = np.asarray(Y[0:80 * total_l / 100], dtype=np.int)

        self.X_val = np.asarray(X[80 * total_l / 100:90 * total_l / 100], dtype=np.float32)
        self.Y_val = np.asarray(Y[80 * total_l / 100:90 * total_l / 100], dtype=np.int)

        self.X_test = np.asarray(X[90 * total_l / 100:total_l], dtype=np.float32)
        self.Y_test = np.asarray(Y[90 * total_l / 100:total_l], dtype=np.int)

        if bool == 'y' or bool == 'Y':
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 48, 48)
            self.X_val = self.X_val.reshape(self.X_val.shape[0], 48, 48)
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 48, 48)

        print 'length of x train and y train:', self.X_train.shape, self.Y_train.shape
        print 'data loaded'

    def make_features(self):

        feature_fns = [hog_feature]
        X_train_feats = extract_features(self.X_train, feature_fns, verbose=True)
        X_val_feats = extract_features(self.X_val, feature_fns)
        X_test_feats = extract_features(self.X_test, feature_fns)

        # Preprocessing: Subtract the mean feature
        mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
        X_train_feats -= mean_feat
        X_val_feats -= mean_feat
        X_test_feats -= mean_feat

        # Preprocessing: Divide by standard deviation. This ensures that each feature
        # has roughly the same scale.
        std_feat = np.std(X_train_feats, axis=0, keepdims=True)
        X_train_feats /= std_feat
        X_val_feats /= std_feat
        X_test_feats /= std_feat

        # Preprocessing: Add a bias dimension
        self.X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
        self.X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
        self.X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

        print self.X_train_feats.shape

        print 'feature extraction done'
        # print X_train_feats[0]

        # print X_val_feats[0]

        # print X_test_feats[0]

    def fit(self,model_name,kwargs,bool):
        if model_name == 'svm':
            print 'SVM MODEL'
            self.model = svm.SVC()

        elif model_name == 'rbf':
            print 'RBF Model'
            self.model = RandomForestClassifier(**kwargs)

        print 'fitting the model...'

        if bool == 'y' or bool == 'Y':
            print 'with HOG features'
            self.model.fit(self.X_train_feats, self.Y_train)

        else:

            self.model.fit(self.X_train, self.Y_train)

        print 'model fitting done...'

    def predict(self,bool):

        print 'predicting the model'

        if bool == 'y' or bool == 'Y':
            print 'with HOG features'
            y_pred = self.model.predict(self.X_test_feats)

            y_val_pred = self.model.predict(self.X_val_feats)

        else:

            y_pred = self.model.predict(self.X_test)
            y_val_pred = self.model.predict(self.X_val)

        print 'Test set accuracy: ', (self.Y_test == y_pred).mean()

        print 'Validation set accuracy: ', (self.Y_val == y_val_pred).mean()






