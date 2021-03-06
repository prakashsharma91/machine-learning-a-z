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
    '/home/ps/Code/machine-learning-a-z/02_regression/01_linear_regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)



## Training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


## Precditing the test set result
y_pred = regressor.predict(X_test)


#Visualising the training set results
plt.scatter(X_train, y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train))
plt.title('Salary vs Experience ( Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

## Visualising the test set results
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_test, regressor.predict(X_test))
plt.title('Salary vs Experience ( Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()