import numpy as np
import matplotlib.pyplot as plt


def run(K, d, sigma, T, L, theta_star, Arm_Stream):

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

    prediction, T_ = SPC(T)

    pred_value = (prediction @ theta_star.reshape(d, 1))[0]
    high_value = pred_value
    low_value = pred_value

    for k in range(K):
        a = Arm_Stream[:, k]
        cur_value = (a @ theta_star.reshape(d, 1))[0]
        high_value = max(high_value, cur_value)
        low_value = min(low_value, cur_value)
    return pred_value, low_value, high_value


def varying_K():
    print("SPC error rates")

    N = 500
    d = 5
    sigma = 4

    L = 0.7

    T_list = [10000, 20000, 30000, 50000]
    K_list = [1000, 10000, 50000, 100000]

    record = np.zeros((len(T_list), len(K_list)))
    for t in range(len(T_list)):
        for k in range(len(K_list)):
            T = T_list[t]
            K = K_list[k]
            err_counter = 0.0

            for i in range(N):
                theta_star = 4 * np.random.rand(d)
                Arm_Stream = (
                    0.16 * np.random.rand(d, K) + 0.19)*2/np.sqrt(d)
                opt_arm = 0.319 * np.ones(d)  # 
                Arm_Stream[:, 0] = opt_arm
                print("K=%d, ite=%d/%d" % (K, i+1, N))
                pred_value, low_value, high_value = run(
                    K=K, d=d, sigma=sigma, T=T, L=L, theta_star=theta_star, Arm_Stream=Arm_Stream)
                if high_value-pred_value > 1e-5:
                    err_counter += 1

            print("Experiment with K=", K, " T= ", T,
                  " terminates. The error rate is ", err_counter / N)
            record[t, k] = err_counter / N

    print("T list: ", T_list)
    print("K list: ", K_list)
    print(record)

    


varying_K()
