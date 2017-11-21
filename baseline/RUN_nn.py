import argparse
import datetime
from Neural_network_with_HOG_CH import *
from Neural_network_with_rawPixels import *
from Neural_network_with_rawPixels_BN import *
from Neural_network_with_HOG_CH_batchnorm import *

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', '-lr',help="Pass the learning rate",required='True')
parser.add_argument('--epochs', '-e',help="Pass number of epochs to run",required='True')
parser.add_argument('--RawFeatures', '-rf',help="Pass r for Raw pixels and f for extracting Features",required='True')
parser.add_argument('--BatchNormalization', '-bn',help="Pass true for batch normalization",required='True')

args = parser.parse_args()

learning_rate = float(args.learning_rate)
epochs = int(args.epochs)

start = datetime.datetime.now()

if args.RawFeatures == 'f':

    if args.BatchNormalization == 'true':
        model = Neural_network_BN(learning_rate, epochs)
    else:
        model = Neural_network(learning_rate, epochs)
    model.load_data()
    model.make_features()

else:
    if args.BatchNormalization == 'true':
        model = Neural_network_raw_BN(learning_rate, epochs)
    else:
        model = Neural_network2(learning_rate, epochs)
    model.load_data()

model.fit_predict(512,512)
model.plot_training_loss()

end = datetime.datetime.now()

print 'total time taken is : ', format(end-start)
