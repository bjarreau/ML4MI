import tensorflow as tf
import pandas as pd
import sklearn
import pickle
import sys


method = sys.argv[1]

if method not in ["ANN", "SVM", "DT"]:
    raise ValueError("Please specify model as 'ANN', 'SVM', or 'DT'")

if method == 'ANN':
    model = tf.keras.models.load_model('ANN')
elif method == 'SVM':
    model = pickle.load(open('SVM', 'rb'))
elif method == 'DT':
    model = pickle.load(open('DT', 'rb'))








































