import argparse
import datetime
from Neural_network_with_HOG_CH import *

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', '-lr',help="Pass the learning rate",required='True')
parser.add_argument('--epochs', '-e',help="Pass number of epochs to run",required='True')

args = parser.parse_args()

learning_rate = float(args.learning_rate)
epochs = int(args.epochs)

model = Neural_network(learning_rate,epochs)

start = datetime.datetime.now()
model.load_data()
model.make_features()

model.fit_predict(256,256)
model.plot_training_loss()

end = datetime.datetime.now()

print 'total time taken is : ', format(end-start)
