import numpy as np

from sklearn.linear_model import LinearRegression

# Load the matrix and vector from .npy files
matrix_file_path = r'.\employee_matrix.npy'
vector_file_path = r'.\employee_vector.npy'


X = np.load(matrix_file_path)
y = np.load(vector_file_path)

d = len(X[0, :])

X_normalized = np.zeros((len(X), d))

# Normalize each column of the matrix to the [0, 1] range
for i in range(d):
    col_max = np.max(X[:, i])
    X_normalized[:, i] = X[:, i] / col_max

# Perform linear regression
regressor = LinearRegression()
regressor.fit(X_normalized, y)

# Get the regression parameters (coefficients)
parameters = regressor.coef_
intercept = regressor.intercept_

print("Regression coefficients:", parameters)
print("Intercept:", intercept)
