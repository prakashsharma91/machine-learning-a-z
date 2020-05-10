from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import
dataset = pd.read_csv(
    '/home/ps/Code/machine-learning-a-z/02_regression/04_support_vector_regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Hanlding missing data


# Encoding & Transform
y = y.reshape(len(y), 1)

# feature scaling
sc_X = StandardScaler()  # different because of different magnitude
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Traning the SVR model on the whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y)


# Predicting a new result
y_pred = regressor.predict(sc_X.fit_transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)


# Visualising the SVR Results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(
    regressor.predict(X)), color='blue')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualising the polynomial regression resutls (for higher resultion and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)),
                   max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(
    regressor.predict(sc_X.transform(X_grid))), color='blue')
plt.title('Truth or bluff (Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
