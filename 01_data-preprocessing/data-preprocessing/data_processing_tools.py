from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import data
dataset = pd.read_csv(
    '/home/ps/Code/machine-learning-a-z/data-preprocessing/data-preprocessing/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Missing data handling
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)


# Encoding the independent variable
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # ouput is np.array. why ?
# print(X)


# Encoding the dependent variable
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)


# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
# print(X_train)
#print(X_test)
# print(y_train)
# print(y_test)


# Feature scaling - Needed for data having dominating feature
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
#print(X_train)
#print(X_test)
