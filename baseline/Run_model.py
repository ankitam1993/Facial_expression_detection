import argparse
import datetime
from SVM_RF import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-m',required = True,help="Pass the model name: rbf, SVM, nn")
parser.add_argument('--arguments', '-a',help="Pass the arguments according to the model")
parser.add_argument('--features', '-f',help="Pass the arguments y or n",required = True)

args = parser.parse_args()

model_name = args.model_name
arguments = args.arguments
bool = args.features

print bool
kargs = dict()

if model_name == 'rbf':
    kargs['n_estimators'] = int(arguments)

start = datetime.datetime.now()
model = baseline_models()

model.load_data(bool)

if bool == 'y' or bool == 'Y':
    print 'Extract HOG features'
    model.make_features()

model.fit(model_name,kargs,bool)
model.predict(bool)

end = datetime.datetime.now()
print 'total time taken is : ', format(end-start)


