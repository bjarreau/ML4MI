import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import pickle
import sys
from os import listdir
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn import metrics

###Adding command line tags for more argument transparency
parser = argparse.ArgumentParser("Build, test, and use ML models")
parser.add_argument("-m", "--model", help="Key for model to run: DT, ANN, SVM")
parser.add_argument("-i", "--input", help="Path to input csv file")
parser.add_argument("-o", "--output", help="Output directory which holds models and metrics")
args = parser.parse_args()    
                    

method, inp, output = args.model, args.input, args.output
files = listdir()

#Ensure model provided is available for this program
if method not in ["ANN", "SVM", "DT"]:
    raise ValueError("Please specify model as 'ANN', 'SVM', or 'DT'")

model_loc = f"{output}/MODEL/"    

#Load model based on --model tag
if method == 'ANN':
    model = tf.keras.models.load_model(f"{model_loc}/ANN")
elif method == 'SVM':
    model = pickle.load(open(f"{model_loc}/SVM.sav", 'rb'))
elif method == 'DT':
    model = pickle.load(open(f"{model_loc}/{decision_tree.sav}", 'rb'))

##Read in input and predict with model
inputs = pd.read_csv(inp)   
X_test = inputs.drop(["Label"], axis=1)
y_test = inputs["Label"]
prediction = model.predict(X_test)

#Tranform ANN output to classes to match other models
if method == 'ANN':
    classes = np.array([0, 1])
    output = np.array([*map(np.argmax, prediction)])
    
def get_scores(acc, labels, prediction):
    f1 = metrics.f1_score(labels, prediction)
    conf_matrix = metrics.confusion_matrix(labels, prediction)
    sens = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    spec = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    auc = metrics.roc_auc_score(labels, prediction)
    add_plots(f1, sens, spec, auc, acc)

def add_plots(f1, sens, spec, auc, acc):
  plt.bar(1, sens, label="{} Sensitivity".format(method))
  plt.bar(2, spec, label="{} Specificity".format(method))
  plt.bar(3, auc, label="{} AUC".format(method))
  plt.bar(4, f1, label="{} F1".format(method))
  plt.bar(5, acc, label="{} ACC".format(method))
  plt.savefig("{}/metrics/{}_saved_run_metrics.png".format(output, method))


acc = accuracy_score(y_test, prediction)
get_scores(acc, y_test, prediction)




































