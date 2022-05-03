import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import pickle
import sys
from os import listdir
import ArgumentParser
import math
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import sklearn.model_selection as model_selection
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn import metrics

###Adding command line tags for more argument transparency
parser = argparse.ArgumentParser("Build, test, and use ML models")
parser.add_argument("-m", "--model")
parser.add_argument("-i", "--input")
parser.add_argument("-o", "--output")
parser.add_argument("-c", "--criterion")
args = parser.parse_args()    
                    
method, inp, out, criterion = args.model, args.input, args.output, args.criterion
files = listdir()

def get_scores(acc, labels, prediction):
    f1 = metrics.f1_score(labels, prediction)
    conf_matrix = metrics.confusion_matrix(labels, prediction)
    sens = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    spec = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    auc = metrics.roc_auc_score(labels, prediction)
    return sens, spec, f1, auc, acc

def add_plots(f1, sens, spec, auc, title):
  plt.bar(1, sens, label="{} Sensitivity".format(title))
  plt.bar(2, spec, label="{} Specificity".format(title))
  plt.bar(3, auc, label="{} AUC".format(title))
  plt.bar(4, f1, label="{} F1".format(title))
  plt.save_fig("{}/metrics/{}_metrics.png".format(output, model))

#Ensure model provided is available for this program
if method not in ["ANN", "SVM", "DT"]:
    raise ValueError("Please specify model as 'ANN', 'SVM', or 'DT'")

model_loc = f"MODEL/{method}"    

data = pd.read_csv(inp)
data = data.dropna()

#Load model based on --model tag
if method == 'ANN':
    model = tf.keras.models.load_model(model_loc)
elif method == 'SVM':
    model = pickle.load(open(model_loc, 'rb'))
elif method == 'DT':
    le = preprocessing.LabelEncoder()
    for attribute in data.columns:
        data[attribute] = le.fit_transform(data[attribute].values)
    samples = data.drop(["Label"], axis=1)
    labels = data.Label

    print("Entropy Measures: ") 
    tree = DecisionTreeClassifier(criterion = criterion).fit(X_train, y_train)  
    prediction = tree.predict(X_test)   
    acc = tree.score(X_test, y_test)
    sens, spec, f1, AUC, acc = get_scores(acc, y_test, prediction)
    e_values = [sens, spec, f1, AUC, acc]
    tree = plot_tree(tree, max_depth=5, feature_names=samples.columns[:-1], class_names=['0','1'], filled=True)
    plt.save_fig("{}/metrics/decision_tree.png".format(output))

##Read in input and predict with model
inputs = pd.read_csv(inp).to_numpy()    
output = model.predict(inputs)

#Tranform ANN output to classes to match other models
if method == 'ANN':
    classes = np.array([0, 1])
    output = np.array([*map(np.argmax, output)])

print(output)




































