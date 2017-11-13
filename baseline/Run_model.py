import argparse
import datetime
from baseline_models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-m',required = True,help="Pass the model name: rbf, SVM")
parser.add_argument('--arguments', '-a',help="Pass the arguments according to the model")

args = parser.parse_args()

model_name = args.model_name
arguments = args.arguments

kargs = dict()

if model_name == 'rbf':
    kargs['n_estimators'] = int(arguments)

model = baseline_models()

model.load_data()
model.fit(model_name,kargs)
model.predict()
