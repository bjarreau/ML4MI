import numpy as np
import pandas as pd
import math
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from seaborn import pairplot

data = pd.read_csv("./Acoustic_Probing_DT.csv")
data = data.drop(["ID", "Trial"], axis=1)
data = data.drop(data.loc[:, data.nunique() == 1], axis=1)

#When the csv reads in, it detects an extra row of NaN
data = data.dropna()

#DecisionTreeClassifier works on numeric only, need to encode
le = preprocessing.LabelEncoder()
for attribute in data.columns:
    data[attribute] = le.fit_transform(data[attribute].values)

display(data)

fig, axes = plt.subplots(5, 6, figsize=(20, 10), sharey=True)

index = 0
for i in range(len(data.keys())):
    key = data.keys()[i]
    if data[key].dtype != 'object':
        sns.histplot(ax=axes[int(index/6)][index%6], x=key, data=data, bins=7)
        index += 1
plt.subplots_adjust(hspace=1)

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

pairplot(data, hue='Label')
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(samples, labels, train_size=0.70,test_size=0.30, random_state=42)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)