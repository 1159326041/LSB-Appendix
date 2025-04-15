import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint


def run(K, d, sigma, eps, T, B, L, theta_star, Arm_Stream, flag):
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
                D[key_list[i]] = num
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
        res = minimize(objective, lam0, method='trust-constr', constraints=[
                       linear_constraint], options={'disp': False, 'maxiter': maxiter}, bounds=bounds)
        lam = res.x
        #
        for i in range(n):
            num = int(lam[i] * budget_record)
            if num > 0.5:
                D[key_list[i]] = num
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
            V = 0
            S = 0

            order = np.random.permutation(K)
            for k in range(K):
                a = Arm_Stream[:, order[k]].reshape(d, 1)
                g_index = tuple(round_arm(a.reshape(d), eps, d))
                if D.get(g_index):
                    D_g = D[g_index]
                    num = D_g
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

    def SPC(T):
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

    prediction, T_ = G_MP_SE(B, T)
    #

    pred_value = (prediction @ theta_star.reshape(d, 1))[0]
    high_value = pred_value
    low_value = pred_value

    for k in range(K):
        a = Arm_Stream[:, k]
        cur_value = (a @ theta_star.reshape(d, 1))[0]
        high_value = max(high_value, cur_value)
        low_value = min(low_value, cur_value)
    return pred_value, low_value, high_value


def varying_B():
    N = 10
    K = 40000
    d = 5
    sigma = 6
    eps = 0.08
    T = 30000
    theta_star = 4 * np.random.rand(d)
    Arm_Stream = (0.16 * np.random.rand(d, K) + 0.19)*2/np.sqrt(d)

    print("G-MP-SE errors with varying B")
    print("Error Bar =", 2 * eps * np.linalg.norm(theta_star))

    data = []

    B_list = [3, 6, 9, 12]
    for B in B_list:
        record = np.zeros(N)
        for i in range(N):
            print("B=%d, ite=%d/%d" % (B, i+1, N))
            pred_value, low_value, high_value = run(
                K=K, d=d, sigma=sigma, eps=eps, T=T, B=B, theta_star=theta_star, Arm_Stream=Arm_Stream)
            record[i] = high_value-pred_value
        data.append(record)

    #
    plt.violinplot(data, B_list, widths=1.6)

    plt.xlabel(r"The number of passes $B$")
    plt.ylabel(r"Prediction error")
    plt.grid(1)
    plt.show()


def varying_T():
    N = 40
    d = 5
    sigma = 5
    eps = 0.15
    B = 5
    L = 0.7
    K = 10000

    data = []

    T_list = [8000,  16000, 32000]
    #

    print("G-MP-SE errors")
    #

    fig, ax = plt.subplots(1, 1)

    for T in T_list:
        #
        #
        flag = False
        record = np.zeros(N)
        for i in range(N):
            theta_star = 4 * np.random.rand(d)
            Arm_Stream = (0.16 * np.random.rand(d, K) + 0.19)*2/np.sqrt(d)
            opt_arm = 0.315 * np.ones(d)  #

            Arm_Stream[:, 0] = opt_arm
            print("T = %d, K = %d, ite = %d/%d" % (T, K, i+1, N))
            pred_value, low_value, high_value = run(
                K=K, d=d, sigma=sigma, eps=eps, T=T, B=B, L=L, theta_star=theta_star, Arm_Stream=Arm_Stream, flag=flag)
            record[i] = high_value-pred_value
        data.append(record)

    #
    ax.boxplot(data,  widths=0.3,  showmeans=True, meanline=True, patch_artist=True,
               meanprops={"color": 'blue',
                          "linestyle": '--', "linewidth": 1.5},
               medianprops={"color": "C0", "linewidth": 1.5},
               boxprops={"facecolor": "C0", "edgecolor": "white",
                         "linewidth": 1.5, "alpha": 0.3},
               whiskerprops={"color": "C0", "linewidth": 1.5},
               capprops={"color": "C0", "linewidth": 1.5}
               )

    ax.set_xlabel(r"Sample Budget $T$")
    ax.set_ylabel(r"Prediction Error $\langle\theta^*, a^*-\hat{a}\rangle$")
    ax.set_xticks(range(1, 1 + len(T_list)))
    ax.set_xticklabels(T_list)

    median_handle = mlines.Line2D([], [], color='C0', label='Median')
    mean_handle = mlines.Line2D(
        [], [], color='blue', linestyle='--', label='Mean')
    ax.legend(handles=[median_handle, mean_handle])

    ax.grid(1)
    plt.show()


#


def varying_T_Gopt_vs_unif():
    N = 40
    d = 5
    sigma = 5
    eps = 0.15
    B = 5
    L = 0.7
    K = 10000

    data1 = []
    data2 = []

    T_list = [8000,  16000, 32000]
    #

    print("G-MP-SE errors")
    #

    fig, ax = plt.subplots(1, 1)

    for T in T_list:
        #
        #
        #
        record1 = np.zeros(N)

        for i in range(N):
            theta_star = 4 * np.random.rand(d)
            Arm_Stream = (0.16 * np.random.rand(d, K) + 0.19)*2/np.sqrt(d)
            opt_arm = 0.315 * np.ones(d)  #

            Arm_Stream[:, 0] = opt_arm
            print("T = %d, K = %d, ite = %d/%d" % (T, K, i+1, N))
            pred_value, low_value, high_value = run(
                K=K, d=d, sigma=sigma, eps=eps, T=T, B=B, L=L, theta_star=theta_star, Arm_Stream=Arm_Stream, flag=True)
            record1[i] = high_value-pred_value

        data1.append(record1)

    #
    ax.boxplot(data1,  widths=0.3,  showmeans=True, meanline=True, patch_artist=True,
               meanprops={"color": 'blue',
                          "linestyle": '--', "linewidth": 1.5},
               medianprops={"color": "C0", "linewidth": 1.5},
               boxprops={"facecolor": "C0", "edgecolor": "white",
                         "linewidth": 1.5, "alpha": 0.3},
               whiskerprops={"color": "C0", "linewidth": 1.5},
               capprops={"color": "C0", "linewidth": 1.5}
               )

    #
    ax.set_xlabel(r"Sample Budget $T$")
    ax.set_ylabel(r"Prediction Error $\langle\theta^*, a^*-\hat{a}\rangle$")
    ax.set_xticks(range(1, 1 + len(T_list)))
    ax.set_xticklabels(T_list)

    median_handle = mlines.Line2D([], [], color='C0', label='Median')
    mean_handle = mlines.Line2D(
        [], [], color='blue', linestyle='--', label='Mean')
    ax.legend(handles=[median_handle, mean_handle])

    ax.grid(1)

    plt.show()


varying_T_Gopt_vs_unif()
