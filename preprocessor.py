import numpy as np
import pandas as pd
import math
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from seaborn import pairplot
import sklearn.model_selection as model_selection

data = pd.read_csv("./RAW/Acoustic_Probing_DT.csv")
data = data.drop(["ID", "Trial"], axis=1)
data = data.drop(data.loc[:, data.nunique() == 1], axis=1)

#When the csv reads in, it detects an extra row of NaN
data = data.dropna()

#DecisionTreeClassifier works on numeric only, need to encode
le = preprocessing.LabelEncoder()
for attribute in data.columns:
    data[attribute] = le.fit_transform(data[attribute].values)

#correlation matrix using heatmap.
corelationBetweenCols = data.corr();
plt.figure(figsize=(20,20))
sns.heatmap(corelationBetweenCols, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Correlation between different features')

data = data.drop(["Tx Phase at -5", "Tx Phase at 2"], axis=1) # reduce highly correlated fields
data = data.drop(["Tx Amplitude at -5", "Tx Amplitude at -4", "Tx Amplitude at -1", "Tx Amplitude at 2", "Tx Amplitude at 4", "Tx Phase at -6", "Tx Phase at -3", "Tx Phase at 1", "Tx Phase at 3", "Tx Phase at 4"], axis=1) # drop fields that are not correlated to the label, they won't help

corelationBetweenCols = data.corr();
plt.figure(figsize=(20,20))
sns.heatmap(corelationBetweenCols, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Correlation between different features')

samples = data.drop(["Label"], axis=1)
labels = data.Label

X_train, X_test, y_train, y_test = model_selection.train_test_split(samples, labels, train_size=0.70,test_size=0.30, random_state=42)
X_train['Label'] = pd.Series(y_train, index=X_train.index)
X_test['Label'] = pd.Series(y_test, index=X_test.index)

X_train.to_csv(".\INPUT\DT\TRAIN\X_train.csv", index=False)
X_test.to_csv(".\INPUT\DT\TEST\X_test.csv", index=False)

data = pd.read_csv("./RAW/Acoustic_Probing.csv")
data = data.drop(["ID", "Trial"], axis=1)
data = data.drop(data.loc[:, data.nunique() == 1], axis=1)

#When the csv reads in, it detects an extra row of NaN
data = data.dropna()

#DecisionTreeClassifier works on numeric only, need to encode
le = preprocessing.LabelEncoder()
for attribute in data.columns:
    data[attribute] = le.fit_transform(data[attribute].values)

#correlation matrix using heatmap.
corelationBetweenCols = data.corr();
plt.figure(figsize=(20,20))
sns.heatmap(corelationBetweenCols, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Correlation between different features')

data = data.drop(["Tx Amplitude at -2.5", "Tx Amplitude at -2", "Tx Amplitude at -1", "Tx Amplitude at 0", "Tx Phase at -2.5", "Tx Phase at -1.5", "Tx Phase at -1"], axis=1) # reduce highly correlated fields
# drop fields that are not correlated to the label, this task is expensive so we are dropping anything under 50%
data = data.drop(["Tx Amplitude at -7", "Tx Amplitude at -4", "Tx Amplitude at -6", "Tx Amplitude at 3", "Tx Amplitude at 4", "Tx Phase at -7", "Tx Phase at -3", "Tx Phase at -0.5", "Tx Phase at 0", "Tx Phase at 1", "Tx Phase at 2", "Tx Phase at 3", "Tx Phase at 4"], axis=1) 

corelationBetweenCols = data.corr();
plt.figure(figsize=(20,20))
sns.heatmap(corelationBetweenCols, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Correlation between different features')

#Accept a pandas dataframe pInput and a set of Training, Validation, and Test ratios
#Randomly sample dataframe without replacement to create the datasets
def getData(pInput, pRatio):
    #first we shuffle the data
    shuffle = pInput.sample(frac=1, random_state=42)

    #Then we create indices to split on
    indices = [int(pRatio[0]*len(shuffle)), int((pRatio[0] + pRatio[1])*len(shuffle))]

    #Now we split
    dTrn, dVal, dTst =  np.split(shuffle, indices)
    return dTrn, dVal, dTst

X_train, X_validation, X_test = getData(data, (0.6, 0.2, 0.2))
X_train.to_csv(".\INPUT\TRAIN\X_train.csv", index=False)
X_test.to_csv(".\INPUT\TEST\X_test.csv", index=False)
X_validation.to_csv(".\INPUT\VALIDATION\X_validation.csv", index=False)