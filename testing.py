import numpy as np

# Define your input data and target values
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
y = np.array([2, 4, 6, 8])

# Compute the least squares solution using the normal equations
X_T = X.transpose()
X_T_X = np.dot(X_T, X)
X_T_X_inv = np.linalg.inv(X_T_X)
X_T_y = np.dot(X_T, y)
beta_hat = np.dot(X_T_X_inv, X_T_y)

# Print the coefficients of the linear model
print('Intercept:', beta_hat[0])
print('Coefficients:', beta_hat[1:])