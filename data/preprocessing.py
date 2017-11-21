import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

def load_data():
    X = []
    Y = []

    print ('loading data')

    with open('../data/fer2013/fer2013.csv', 'r') as f:
        data = csv.reader(f)

        for row in data:

            if row[0] != 'emotion':
                Y.append(int(row[0]))
                z = list(row[1:-1][0].split())
                z = [int(a) for a in z]
                X.append(z)
            else:
                print (row)
        print (np.asarray(X[:10]))
        print (np.asarray(Y[:10]))
        data_to_save = np.column_stack((np.asarray(X),np.asarray(Y)))
        proj_data = TemporaryFile()
        np.save('../data/fer2013/proj_data', data_to_save)
        proj_data.seek(0)
        f.close()
    total_l = len(X)



def main():
    load_data()

if __name__ == '__main__':
    main()