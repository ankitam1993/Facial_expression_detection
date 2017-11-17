from __future__ import division
from __future__ import print_function
import argparse
import math
import numpy as np
from utils import load_data
import tensorflow as tf
from collections import defaultdict
import csv

np.random.seed(0)


class NN(object):
    """A network architecture of simultaneous localization and
       classification of objects in images.

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """
    def __init__(self, alpha=.5, epochs=5):
        self.alpha = alpha
        self.epochs = epochs
        self.x = tf.placeholder(tf.float32, shape=(None, 3600))
        self.y_class = tf.placeholder(tf.float32, shape=(None, 1))
        #self.y_loc = tf.placeholder(tf.float32, shape=(None, 2))

        self.w1 = tf.get_variable("w1", shape = (3600, 256), initializer = tf.contrib.layers.xavier_initializer())
        self.b1 = tf.get_variable("b1", shape = (256,), initializer = tf.contrib.layers.xavier_initializer())
        self.w2 = tf.get_variable("w2", shape=(256, 64), initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = tf.get_variable("b2", shape=(64,), initializer=tf.contrib.layers.xavier_initializer())
        self.w3 = tf.get_variable("w3", shape=(64, 32), initializer=tf.contrib.layers.xavier_initializer())
        self.b3 = tf.get_variable("b3", shape=(32,), initializer=tf.contrib.layers.xavier_initializer())
        self.w4 = tf.get_variable("w4", shape=(32, 2), initializer=tf.contrib.layers.xavier_initializer())
        self.b4 = tf.get_variable("b4", shape=(2,), initializer=tf.contrib.layers.xavier_initializer())
        self.w5 = tf.get_variable("w5", shape=(32, 1), initializer=tf.contrib.layers.xavier_initializer())
        self.b5 = tf.get_variable("b5", shape=(1,), initializer=tf.contrib.layers.xavier_initializer())


        hidden_1_layer = tf.nn.relu(tf.add(tf.matmul(self.x, self.w1), self.b1))
        hidden_2_layer = tf.nn.relu(tf.add(tf.matmul(hidden_1_layer, self.w2), self.b2))
        hidden_3_layer = tf.nn.relu(tf.add(tf.matmul(hidden_2_layer, self.w3), self.b3))
        #self.y_loc_pred = tf.add(tf.matmul(hidden_3_layer, self.w4), self.b4)
        self.y_class_pred = tf.add(tf.matmul(hidden_3_layer, self.w5), self.b5)

        self.class_prediction = tf.to_int32(self.y_class_pred > 0)
        #self.accuracy = tf.reduce_mean(self.class_prediction == self.y_class)
        self.accuracy = tf.metrics.accuracy(self.y_class, self.class_prediction)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_class_pred,labels=self.y_class)
        self.cross_entropy_loss = cross_entropy
        #self.localization_loss =  tf.reduce_sum( tf.squared_difference(self.y_loc, self.y_loc_pred), axis=1)
        self.loss = self.alpha * tf.reduce_sum(self.cross_entropy_loss) + (1 - self.alpha) * tf.reduce_sum(self.localization_loss)
        self.mean_loss = self.alpha * tf.reduce_mean(self.cross_entropy_loss) + (1 - self.alpha) * tf.reduce_mean(self.localization_loss)

        optimizer = tf.train.AdamOptimizer()
        self.updates = optimizer.minimize(self.mean_loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def objective(self, X, y_class, y_loc):
        """Objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the objects.
                This is the coordinate of the top-left corner of the 28x28
                region where the object presents.

        Returns:
            Composite objective function value.
        """

        N, D = X.shape[0], X.shape[1]

        values = {self.x: X, self.y_loc : y_loc, self.y_class : y_class[:, None]}

        return self.sess.run(self.loss, feed_dict = values)

    def predict(self, X):
        """Predict class labels and object locations for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Input matrix where each row is a feature vector.

        Returns:
            y_class (numpy ndarray, shape = (samples,)):
                predicted labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                The predicted (vertical, horizontal) locations of the
                objects.
        """
        # y_predict = []
        # offset_predict = []

        y_predict = np.zeros((X.shape[0],))
        offset_predict = np.zeros([X.shape[0], 2])
        values = {self.x : X}
        y_predict = self.sess.run(self.class_prediction, feed_dict = values).squeeze()
        print (y_predict.shape)
        #y_loc_pred = self.sess.run(self.y_loc_pred, feed_dict=values)

        #offset_predict += y_loc_pred
        return y_predict

    def fit(self, X, y_class, y_loc):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the
                objects.
        """


        N, D = X.shape
        batch_size = 16

        for epoch in range(self.epochs):
            random_ids = np.arange(N)
            np.random.shuffle(random_ids)
            X_shuffled = X[random_ids,:]
            y_class_shuffled = y_class[random_ids]
            y_loc_shuffled = y_loc[random_ids,:]
            mini_batches = [(X_shuffled[i:i + batch_size, :], y_class_shuffled[i:i + batch_size], y_loc_shuffled[i:i + batch_size]) for
                            i in range(0, N, batch_size)]
            for mini_batch in mini_batches:
                X_batch = mini_batch[0]
                y_batch_class = mini_batch[1]
                y_batch_loc = mini_batch[2]
                values = {self.x: X_batch, self.y_class: y_batch_class[:, None] , self.y_loc: y_batch_loc}
                self.sess.run([self.updates], feed_dict=values)

            values = {self.x: X, self.y_loc: y_loc, self.y_class: y_class[:, None]}
            epoch_loss = self.sess.run([self.mean_loss], feed_dict=values)
            print(self.sess.run([tf.reduce_mean(self.cross_entropy_loss), tf.reduce_mean(self.localization_loss)], feed_dict=values))
            print (epoch_loss)




