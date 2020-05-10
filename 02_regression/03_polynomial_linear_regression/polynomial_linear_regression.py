from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
    '/home/ps/Code/machine-learning-a-z/02_regression/03_polynomial_linear_regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the linear regression model on the whole dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Training the polynomial regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


# Visualising the linear regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualising the polynomial regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynominal Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualising the polynomial regression resutls (for higher resultion and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg2.predict(
    poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or bluff (Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result
print(lin_reg.predict([[6.5]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))