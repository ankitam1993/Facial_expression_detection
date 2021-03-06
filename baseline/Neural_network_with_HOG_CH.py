import csv
from extract_features import *
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
#from skimage.feature import hog
import matplotlib.pyplot as plt

# This class is implementing 2-layer Neural network with HOG + Color histogram features ( taken from assignment 1)
class Neural_network(object):

    def __init__(self,learning_rate,epochs):

        self.X_train = []
        self.Y_train = []

        self.X_val = []
        self.Y_val = []

        self.X_test = []
        self.Y_test = []

        self.epochs = epochs
        self.learning_rate = learning_rate

    def load_data(self):


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

        self.X_train = np.asarray(X[0:80*total_l/100],dtype=np.float32)
        self.X_train = self.X_train .reshape(self.X_train.shape[0],48,48)

        self.Y_train = np.asarray(Y[0:80*total_l/100],dtype=np.int)

        self.X_val = np.asarray(X[80*total_l/100:90*total_l/100],dtype=np.float32)
        self.X_val = self.X_val.reshape(self.X_val.shape[0], 48, 48)

        self.Y_val = np.asarray(Y[80*total_l/100:90*total_l/100],dtype=np.int)

        self.X_test = np.asarray(X[90*total_l/100:total_l],dtype=np.float32)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 48, 48)

        self.Y_test = np.asarray(Y[90*total_l/100:total_l],dtype=np.int)

        print 'length of x train and y train:' , self.X_train.shape , self.Y_train.shape
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
        #print X_train_feats[0]

        #print X_val_feats[0]

        #print X_test_feats[0]


    def fit_predict(self,hidden_size_1,hidden_size_2):

        self.input_size = self.X_train_feats.shape[1]
        self.output_size = 7

        self.epoch_loss = []
        self.batch_size = 512

        graph = tf.Graph()

        with graph.as_default():

            # placeholders for input vector
            x = tf.placeholder(tf.float32, [None,self.input_size])
            y_class = tf.placeholder(tf.float32, shape=(None, self.output_size))

            self.w1 = tf.get_variable("w1", shape=(self.input_size, hidden_size_1),
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable("b1", shape=(hidden_size_1,), initializer=tf.contrib.layers.xavier_initializer())
            self.w2 = tf.get_variable("w2", shape=(hidden_size_1, hidden_size_2),
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable("b2", shape=(hidden_size_2,), initializer=tf.contrib.layers.xavier_initializer())
            self.w3 = tf.get_variable("w3", shape=(hidden_size_2, self.output_size),
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b3 = tf.get_variable("b3", shape=(self.output_size,),
                                      initializer=tf.contrib.layers.xavier_initializer())


            hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.w1), self.b1))
            hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1, self.w2), self.b2))
            output_layer = tf.add(tf.matmul(hidden_layer_2, self.w3), self.b3)

            mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y_class))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(mean_loss)
            init = tf.global_variables_initializer()


        with tf.Session(graph=graph) as sess:
            # create initialized variables
            sess.run(init)
            print 'Initialized'

            self.epoch_losses = []

            N = self.X_train_feats.shape[0]

            for epoch in range(self.epochs):
                epoch_loss = 0
                total_batches = int(self.X_train_feats.shape[0] / self.batch_size)

                random_ids = np.arange(N)
                np.random.shuffle(random_ids)
                X_shuffled = self.X_train_feats[random_ids, :]
                y_class_shuffled = self.Y_train[random_ids]

                mini_batches = [(X_shuffled[i:i + self.batch_size, :], y_class_shuffled[i:i + self.batch_size]) for
                                i in range(0, N, self.batch_size)]

                for mini_batch in mini_batches:

                    X_batch = mini_batch[0]
                    Y_batch = self.labels_to_one_hot(mini_batch[1])

                    _, c = sess.run([optimizer, mean_loss], feed_dict={x: X_batch, y_class: Y_batch})

                    epoch_loss += c / total_batches

                self.epoch_losses.append(epoch_loss)

                print "Training Epoch:", (epoch + 1), "loss =", "{:.5f}".format(epoch_loss)

            print "\nTraining complete!"

            # find predictions on val set
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_class, 1))
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

            print "Validation Accuracy:", accuracy.eval(
                {x: self.X_val_feats.reshape(-1, self.input_size), y_class: self.labels_to_one_hot(self.Y_val)})


            print "Test Accuracy:", accuracy.eval(
                {x: self.X_test_feats.reshape(-1, self.input_size), y_class: self.labels_to_one_hot(self.Y_test)})


    def labels_to_one_hot(self,Y,num_classes=7):

        """Convert class labels from scalars to one-hot vectors"""

        num_labels = Y.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + Y.ravel()] = 1

        return labels_one_hot

    def plot_training_loss(self):

        x = np.arange(0, self.epochs, 1)
        plt.plot(x, self.epoch_losses)
        plt.xlabel('iterations')
        plt.ylabel('training loss history')
        plt.savefig('training_loss_%s.jpg'% (str(self.learning_rate)))
        plt.show()


    # We can try this after previous one is implemented successfully
    def other_hog(self):
        fd, hog_image = hog(self.X_train, orientations=8, pixels_per_cell=(32, 32),
                                cells_per_block=(1, 1))

        print 'other hog', fd.shape