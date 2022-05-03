import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import pickle
import sys
import os
from os import listdir
import argparse
import math
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC # "Support vector classifier"
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import sklearn.model_selection as model_selection
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn import metrics

###Adding command line tags for more argument transparency
parser = argparse.ArgumentParser("Build, test, and use ML models")
parser.add_argument("-m", "--model", help="Key for model to run: DT, ANN, SVM")
parser.add_argument("-i", "--input", help="Path to directory with Train/Test dirs")
parser.add_argument("-o", "--output", help="Path to place output artifacts, needs METRICS and MODEL subdirs")
parser.add_argument("-c", "--criteria", help="criteria to use for Decision Tree")
parser.add_argument("-d", "--depth", help="maximum depth for Decision Tree")
parser.add_argument("-k", "--kernel", help="kernel type for SVM: linear, poly, rbf, sigmoid, precomputed")
parser.add_argument("-r", "--regularization", help="regularization amount for SVM (must be positive)")
args = parser.parse_args()    
                    
method, inp, out, criteria, depth, kernel, regularization = args.model, args.input, args.output, args.criteria, args.depth, args.kernel, args.regularization
files = listdir()

if criteria is None:
    criteria = "entropy"
    
if kernel is None:
    kernel = "linear"
    
if regularization is None:
    regularization = 1.0
    
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
  plt.savefig("{}/metrics/{}_metrics.png".format(out, method))
  
def decision_function(model, ax = None, plot_support = True):
   if ax is None:
      ax = plt.gca()
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()

#Ensure model provided is available for this program
if method not in ["ANN", "SVM", "DT"]:
    raise ValueError("Please specify model as 'ANN', 'SVM', or 'DT'")

model_loc = f"MODEL/{method}"    

for filename in os.listdir(inp):
    if filename == "TRAIN":
        for file in os.listdir(f"{inp}/{filename}"):
            if file.endswith(".csv"):
                train = f"{inp}/{filename}/{file}"
    if filename == "TEST":
        for file in os.listdir(f"{inp}/{filename}"):
            if file.endswith(".csv"):
                test = f"{inp}/{filename}/{file}"
                
train_data = pd.read_csv(train)
test_data = pd.read_csv(test)

X_train = train_data.drop(["Label"], axis=1)
y_train = train_data["Label"]

X_test = test_data.drop(["Label"], axis=1)
y_test = test_data["Label"]

#Load model based on --model tag
if method == 'ANN':
    model = tf.keras.models.load_model(model_loc)
    #Tranform ANN output to classes to match other models:
    classes = np.array([0, 1])
    output = np.array([*map(np.argmax, output)])
elif method == 'SVM':
    clf = SVC(kernel=f'{kernel}', C=float(regularization))
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    get_scores(acc, y_test, predictions)
    
    fig, axes = plt.subplots(len(X_train.keys()) - 1, len(X_train.keys()) - 1, figsize=(40, 40), sharey=False)

    w = clf.coef_[0]           # w consists of 10 elements
    b = clf.intercept_[0]      # b consists of 1 element

    for i in range(len(X_train.keys()) - 1):
      key1 = X_train.keys()[i]
      for j in range(len(X_train.keys()) - 1):
        key2 = X_train.keys()[j]
        if key1 != key2:
          h = 0.2
          x_points = np.linspace(X_train[key1].min(), X_train[key1].max())    # generating x-points from -1 to 1
          y_points = -(w[i] / w[j]) * x_points - b / w[i]  # getting corresponding y-points
          axes[i][j].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
          axes[i][j].scatter(X_train[key1], X_train[key2], c=train_data["Label"].map({1:'red', 0:'blue'}))
          axes[i][j].plot(x_points, y_points, c='r')
          axes[i][j].set_xlabel(key1)
          axes[i][j].set_ylabel(key2)

    plt.subplots_adjust(hspace=1)
    plt.savefig("{}/METRICS/SVM.png".format(out), format='png', bbox_inches = "tight")
    
    pickle.dump(clf, open("{}/MODEL/SVM.sav".format(out), 'wb'))

elif method == 'DT':
    tree = DecisionTreeClassifier(criterion = criteria).fit(X_train, y_train)  
    prediction = tree.predict(X_test)   
    acc = tree.score(X_test, y_test)
    get_scores(acc, y_test, prediction)
    
    _ = plot_tree(tree, max_depth=int(depth), feature_names=X_train.columns[:-1], class_names=['0','1'], filled=True)
    plt.savefig("{}/METRICS/decision_tree.png".format(out), format='png', bbox_inches = "tight")
    
    pickle.dump(tree, open("{}/MODEL/decision_tree.sav".format(out), 'wb'))





































