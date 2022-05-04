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
from tensorflow import keras 
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation 
from tensorflow.keras.optimizers import Adam

###Adding command line tags for more argument transparency
parser = argparse.ArgumentParser("Build, test, and use ML models")
parser.add_argument("-m", "--model", help="Key for model to run: DT, ANN, SVM")
parser.add_argument("-i", "--input", help="Path to directory with Train/Test dirs")
parser.add_argument("-o", "--output", help="Path to place output artifacts, needs METRICS and MODEL subdirs")
parser.add_argument("-c", "--criteria", help="criteria to use for Decision Tree")
parser.add_argument("-d", "--depth", help="maximum depth for Decision Tree")
parser.add_argument("-k", "--kernel", help="kernel type for SVM: linear, poly, rbf, sigmoid, precomputed")
parser.add_argument("-r", "--regularization", help="regularization amount for SVM (must be positive)")
parser.add_argument("-s", "--earlystop", help="Use early stop")
parser.add_argument("-a", "--activation", help="activation function for hidden layers: relu, sigmoid, tanh")
parser.add_argument("-l", "--learningrate", help="learning rate for optimizer")
args = parser.parse_args()    
                    
method, inp, out, criteria, depth, kernel, regularization, earlystop, activation, learningrate = args.model, args.input, args.output, args.criteria, args.depth, args.kernel, args.regularization, args.earlystop, args.activation, args.learningrate
files = listdir()

if criteria is None:
    criteria = "entropy"
    
if kernel is None:
    kernel = "linear"
    
if regularization is None:
    regularization = 1.0
    
if earlystop is None:
    earlystop = False
else:
    earlystop = True

if activation is None:
    activation = "relu"

if learningrate is None:
    learningrate = 0.0001
    
def get_scores(acc, labels, prediction, title):
    f1 = metrics.f1_score(labels, prediction)
    conf_matrix = metrics.confusion_matrix(labels, prediction)
    sens = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    spec = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    auc = metrics.roc_auc_score(labels, prediction)
    add_plots(f1, sens, spec, auc, acc, title)

def add_plots(f1, sens, spec, auc, acc, title):
  plt.clf()
  plt.bar(1, sens, label="{} Sensitivity".format(method))
  plt.bar(2, spec, label="{} Specificity".format(method))
  plt.bar(3, auc, label="{} AUC".format(method))
  plt.bar(4, f1, label="{} F1".format(method))
  plt.bar(5, acc, label="{} ACC".format(method))
  plt.savefig("{}/metrics/{}_{}_metrics.png".format(out, method, title))
  
def decision_function(model, ax = None, plot_support = True):
   if ax is None:
      ax = plt.gca()
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()

#Ensure model provided is available for this program
if method not in ["ANN", "SVM", "DT"]:
    raise ValueError("Please specify model as 'ANN', 'SVM', or 'DT'")

model_loc = f"MODEL/{method}"    

def read_csv(path):
    for file in os.listdir(path):
        if file.endswith(".csv"):
            data = pd.read_csv(f"{path}/{file}")
    xs = data.drop(["Label"], axis=1)
    ys = data["Label"]
    return (xs, ys)
    
def predict_model(predicted, labels, input):
    classes = sorted(np.unique(labels))
    prediction = classes[np.argmax(inputr)]
    return prediction
    
def summarize_diagnostics(history):
    plt.clf()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.savefig("{}/metrics/{}_accuracy.png".format(out, method))

    plt.clf()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.savefig("{}/metrics/{}_loss.png".format(out, method))

for filename in os.listdir(inp):
    if filename == "TRAIN":
        X_train, y_train = read_csv(f"{inp}/{filename}")
    elif filename == "TEST":
        X_test, y_test = read_csv(f"{inp}/{filename}")
    elif filename == "VALIDATION":
        X_validation, y_validation = read_csv(f"{inp}/{filename}")        

#Load model based on --model tag
if method == 'ANN':
    model = Sequential([
        Dense(20, input_dim=10, activation=f'{activation}'),
        Dense(20, activation=f'{activation}'),
        Dense(2, activation='softmax')
    ])

    model.compile(Adam(lr=float(learningrate)), loss="sparse_categorical_crossentropy", metrics=['accuracy'])   
    model.summary()

    if earlystop:
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=0.001)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[callback], epochs=100, batch_size=64)
    else:
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)
    
    predictions = model.predict(X_test)
    predictions=np.argmax(predictions,axis=1)
    acc = accuracy_score(y_test, predictions)
    get_scores(acc, y_validation, predictions, "test")
    summarize_diagnostics(history)

    predictions = model.predict(X_validation)
    predictions=np.argmax(predictions,axis=1)
    acc = accuracy_score(y_validation, predictions)
    get_scores(acc, y_validation, predictions, "validation")
elif method == 'SVM':
    clf = SVC(kernel=f'{kernel}', C=float(regularization))
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    get_scores(acc, y_test, predictions, "test")
    
    predictions = clf.predict(X_validation)
    acc = accuracy_score(y_validation, predictions)
    get_scores(acc, y_validation, predictions, "validation")
    
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
          axes[i][j].scatter(X_train[key1], X_train[key2], c=y_train.map({1:'red', 0:'blue'}))
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
    get_scores(acc, y_test, prediction, "test")
    
    _ = plot_tree(tree, max_depth=int(depth), feature_names=X_train.columns[:-1], class_names=['0','1'], filled=True)
    plt.savefig("{}/METRICS/decision_tree.png".format(out), format='png', bbox_inches = "tight")
    
    pickle.dump(tree, open("{}/MODEL/decision_tree.sav".format(out), 'wb'))





































