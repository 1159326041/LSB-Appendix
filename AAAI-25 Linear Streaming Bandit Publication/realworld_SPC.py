import numpy as np
import matplotlib.pyplot as plt

# Load the matrix from .npy files
matrix_file_path = r'.\data\employee_matrix.npy'
X = np.load(matrix_file_path)
d = len(X[0, :])
#
#


# The true regression parameter is computed by running data_linear_regression.py
theta_star = np.array([-0.02692076,  0.02658687, -0.69410548,  0.10087781,  0.34861592,  0.90063385,
                       -0.02649656,  0.36872516])
Arm_Stream = np.zeros((len(X), d))


# Normalize each column of the matrix to the [0, 1] range
for i in range(d):
    col_max = np.max(X[:, i])
    Arm_Stream[:, i] = X[:, i] / col_max

Arm_Stream = Arm_Stream.T


# Set up the parameters for the involved algorithms

N = 500
L = 1
sigma = 0.15

# Define the algorithms


def SPC(K, T):
    # parameters
    eta = 1
    precision = np.sqrt(2 * d * np.log(1 + T*L*L/(eta * d)) / T) * 0.7

    # begin scanning
    V = eta * np.diag(np.ones(d))
    S = 0
    t = 1
    a_hat = None

    order = np.random.permutation(K)
    for k in range(K):
        a = Arm_Stream[:, order[k]]
        a_reshape = a.reshape(d, 1)
        while a @ np.linalg.inv(V) @ a >= precision**2 and t <= T:
            r_t = (theta_star @ a) + sigma * np.random.randn()
            V += a_reshape @ a_reshape.T
            S += r_t * a_reshape
            t += 1
        theta_hat = (np.linalg.inv(V) @ S).reshape(d)
        if a_hat is None or theta_hat @ a_hat < theta_hat @ a:
            a_hat = a

    return a_hat, t - 1


K_list = [5000, 10000, 15000]
T_list = [1000, 3000, 5000, 8000]

# Figure details
#

error_rate = np.zeros((len(K_list), len(T_list)))

for k in range(len(K_list)):
    K = K_list[k]

    # Compute the highest sample mean
    high_value = -1e7
    for j in range(K):
        a = Arm_Stream[:, j]
        cur_value = a @ theta_star
        high_value = max(high_value, cur_value)

    for t in range(len(T_list)):
        T = T_list[t]
        cnt = 0.0

        for i in range(N):
            print("T = %d, K = %d, ite = %d/%d" % (T, K, i+1, N))
            a_hat, _ = SPC(K, T)
            error = high_value - a_hat @ theta_star
            if error > 1e-5:
                cnt += 1

        error_rate[k, t] = cnt / N
        print("     error rate = ", cnt / N)
print("T list: ", T_list)
print("K list: ", K_list)
print("Error probability: \n", error_rate)
