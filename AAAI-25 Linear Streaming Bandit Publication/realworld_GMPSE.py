import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint

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


# Set up the parameters
N = 40
eps = 0.08
B = 5


sigma = 0.15

# Defining the algorithms

def run(K, T, B):
    #
    #
    #
    #
    #
    #
    G_eps = {}
    #

    #

    def round_arm(a, eps, d):
        # round arm a \in \mathbb{R}^d to a grid point
        # data size: a -- d, eps -- 1, d -- 1, g -- d
        g = [0 for i in range(d)]
        alpha = np.sqrt(d) / eps
        for i in range(d):
            i_1 = np.floor(a[i] * alpha)
            i_2 = np.ceil(a[i] * alpha)

            if np.abs(i_1 / alpha - a[i]) < np.abs(i_2 / alpha - a[i]):
                g[i] = int(i_1)
            else:
                g[i] = int(i_2)

        # note that g is a list of integers
        return g

    def assign_uniform_active_arms(B, T, C_G):
        D = {}

        budget_pass = int(T/(B-2))
        budget_record = budget_pass

        key_list = list(C_G.keys())
        
        n = len(key_list)

        lam = np.ones(n)/n
        for i in range(n):
            num = int(lam[i] * budget_record) 
            if num > 0.5:
                D[ key_list[i] ]  = num 
                budget_pass -= num 
        
        while budget_pass > 0.5:
            g_index = np.random.choice(n)
            g = key_list[g_index]
            if D.get(g):
                D[g] += 1
            else:
                D[g] = 1
            budget_pass -= 1

        return D

    def assign_G_optimal(B, T, C_G):
        D = {}
        eta = 0.003

        budget_pass = int(T/(B-2))
        budget_record = budget_pass
        key_list = list(C_G.keys())
        
        n = len(key_list)

     
        X = np.zeros((n, d))
        for i in range(n):
            X[i, :] = np.array(key_list[i]) 
        xxT = [X[i, :].reshape(d, 1) @ X[i, :].reshape(1, d) for i in range(n)]
        def objective(lam):
            mat = eta * np.eye(d)
            for i in range(n):
                mat += lam[i] * xxT[i]
            mat = np.linalg.inv(mat)
            lst = [X[i, :].reshape(1, d) @ mat @  X[i, :].reshape(d, 1)
                for i in range(n)]
            
            return np.max(lst)
        lam0 = np.ones(n)/n
        linear_constraint = LinearConstraint([1 for i in range(n)], [1], [1])

        bounds = Bounds(lb=[0.02/n for i in range(n)])

        
        maxiter = 30
        res = minimize(objective, lam0, method='trust-constr', constraints=[linear_constraint], options={'disp': False, 'maxiter': maxiter}, bounds=bounds)
        lam = res.x
        #
        for i in range(n):
            num = int(lam[i] * budget_record) 
            if num > 0.5:
                D[ key_list[i] ]  = num 
                budget_pass -= num 
        
        while budget_pass > 0.5:
            g_index = np.random.choice(n)
            g = key_list[g_index]
            if D.get(g):
                D[g] += 1
            else:
                D[g] = 1
            budget_pass -= 1

        return D

    def G_MP_SE(B, T):
        # an implementation of the G-MP-SE algorithm in the paper
        C_G = {}
        cnt = 0

        # pass 1
        order = np.random.permutation(K)
        for k in range(K):
            g_index = tuple(
                round_arm(Arm_Stream[:, order[k]].reshape(d), eps, d))
            if G_eps.get(g_index):
                G_eps[g_index] += 1
                C_G[g_index] += 1
            else:
                G_eps[g_index] = 1
                C_G[g_index] = 1

        # pass 2 to pass B-1
        for b in range(2, B):
            #
            # 
          
            
            D = assign_G_optimal(B, T, C_G)
            
            V = 0.003 * np.eye(d)
            S = 0

            order = np.random.permutation(K)
            for k in range(K):
                a = Arm_Stream[:, order[k]].reshape(d, 1)
                g_index = tuple(round_arm(a.reshape(d), eps, d))
                if D.get(g_index):
                    D_g = D[g_index]
                    num = D_g# 
                    D_g -= num
                    if D_g < 0.5:
                        D.pop(g_index)
                    else:
                        D[g_index] = D_g
                    # pull this arm num times
                    for n in range(num):
                        r_t = (theta_star @ a)[0] + sigma * np.random.randn()
                        V += a @ a.T
                        S += r_t * a
                        cnt += 1
            
            theta_hat = (np.linalg.inv(V) @ S).reshape(d)
            Empirical_Means = {}
            for g in C_G:
                Empirical_Means[g] = (
                    theta_hat @ (np.array(g)).reshape(d, 1))[0]
            N_elim = int(np.ceil(K**((B-b-0.0)/(B-2))) -
                         np.ceil(K**((B-b-1.0)/(B-2))))

            # arm elimination
            while N_elim > 0.5:
                k = Empirical_Means.keys()
                v = Empirical_Means.values()
                sub_opt_point = min(zip(v, k))[1]
                if C_G[sub_opt_point] > N_elim:
                    C_G[sub_opt_point] -= N_elim
                    N_elim = 0
                else:
                    N_elim -= C_G[sub_opt_point]
                    C_G.pop(sub_opt_point)
                    Empirical_Means.pop(sub_opt_point)

        # pass B
        #
        order = np.random.permutation(K)
        for k in range(K):
            a = Arm_Stream[:, order[k]]
            g_index = tuple(round_arm(a, eps, d))
            if C_G.get(g_index):
                return a, cnt

        return None, cnt


    return G_MP_SE(B, T)
    
T_list = [1000, 3000, 5000]
K_list = [5000, 10000]
#




# Figure details
rows = 1
cols = 2

fig, ax = plt.subplots(rows, cols)

for k in range(len(K_list)):
    K = K_list[k]
    # Compute the highest sample mean
    high_value = -1e7
    for j in range(K):
        a = Arm_Stream[:, j]
        cur_value = a @ theta_star
        high_value = max(high_value, cur_value)

    data = []
    for t in range(len(T_list)):
        T = T_list[t]
        record = []

        for i in range(N):
            print("T = %d, K = %d, ite = %d/%d" % (T, K, i+1, N))
            a_hat, _ = run(K, T, B)
            error = high_value - a_hat @ theta_star
            record.append(error)
        
        data.append(record)

    ax[k].violinplot(data,  widths=0.2, showmeans=True, showextrema=True)


    ax[k].set_title(r"Dataset: Employee's Performance for HR Analytics ($K=%d$)" % (K))


    ax[k].set_xlabel(r"Sample Budget $T$")
    ax[k].set_ylabel(r"Prediction Error $\langle\theta^*, a^*-\hat{a}\rangle$")
    ax[k].set_xticks(range(1, 1 + len(T_list)))
    ax[k].set_xticklabels(T_list)
    ax[k].grid(True)

plt.show()








