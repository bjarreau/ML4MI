import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import pickle
import sys
from os import listdir
import argparse

###Adding command line tags for more argument transparency
parser = argparse.ArgumentParser("Build, test, and use ML models")
parser.add_argument("-m", "--model")
parser.add_argument("-i", "--input")
args = parser.parse_args()    
                    

method, inp = args.model, args.input
files = listdir()

#Ensure model provided is available for this program
if method not in ["ANN", "SVM", "DT"]:
    raise ValueError("Please specify model as 'ANN', 'SVM', or 'DT'")

model_loc = f"MODEL/{method}"    

#Load model based on --model tag
if method == 'ANN':
    model = tf.keras.models.load_model(model_loc)
elif method == 'SVM':
    model = pickle.load(open(model_loc, 'rb'))
elif method == 'DT':
    model = pickle.load(open(model_loc, 'rb'))

##Read in input and predict with model
inputs = pd.read_csv(inp).to_numpy()    
output = model.predict(inputs)

#Tranform ANN output to classes to match other models
if method == 'ANN':
    classes = np.array([0, 1])
    output = np.array([*map(np.argmax, output)])


print(output)




































