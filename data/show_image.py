import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
import os
import scipy.misc

class baseline_models(object):

    def __init__(self):

        self.X_train = []
        self.Y_train = []

        self.X_val = []
        self.Y_val = []

        self.X_test = []
        self.Y_test = []

        self.model = None

    def image_kaggle(self):

        w, h = 48, 48
        image = np.zeros((h, w), dtype=np.uint8)
        id = 1
        with open('./fer2013/fer2013.csv', 'rb') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            headers = datareader.next()
            print headers
            for row in datareader:
                emotion = row[0]
                pixels = map(int, row[1].split())
                usage = row[2]
                print emotion, type(pixels[0]), usage
                pixels_array = np.asarray(pixels)

                image = pixels_array.reshape(w, h)
                print image.shape

                stacked_image = np.dstack((image,) * 3)
                print stacked_image.shape

                image_folder = os.path.join('./images/', usage)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_file = os.path.join(image_folder, str(id) + '.jpg')
                scipy.misc.imsave(image_file, stacked_image)
                id += 1
                if id % 100 == 0:
                    print('Processed {} images'.format(id))

        print("Finished processing {} images".format(id))

    def load_data(self,bool):

        X = []
        Y = []

        print 'loading data'

        with open('./fer2013/fer2013.csv', 'r') as f:
            data = csv.reader(f)

            i = 0

            for row in data:

                if row[0] != 'emotion' and i <= 3:
                    Y.append(int(row[0]))
                    z = list(row[1:-1][0].split())
                    z = [int(a) for a in z]
                    X.append(z)
                else:
                    print row

                if i > 3:
                    break

                i +=1

        total_l = len(X)
        print len(X)
        az = np.asarray(X[1], dtype=np.float32)
        print az.shape
        az = az.reshape((48,48))
        print az.shape
        plt.imshow(az)
        plt.show()
s = baseline_models()
#s.load_data('Y')
s.image_kaggle()