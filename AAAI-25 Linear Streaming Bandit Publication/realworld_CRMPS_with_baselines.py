import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# Load the matrix from .npy files
matrix_file_path = r'./data/employee_matrix.npy'

X = np.load(matrix_file_path)
d = len(X[0, :])
K = 10000


# The true regression parameter is computed by running data_linear_regression.py
theta_star = np.array([-0.02692076,  0.02658687, -0.69410548,  0.10087781,  0.34861592,  0.90063385,
                       -0.02649656,  0.36872516])
Arm_Stream = np.zeros((len(X), d))


# Normalize each column of the matrix to the [0, 1] range
for i in range(d):
    col_max = np.max(X[:, i])
    Arm_Stream[:, i] = X[:, i] / col_max

Arm_Stream = Arm_Stream.T

# Compute the highest sample mean
high_value = -1e7
for k in range(K):
    a = Arm_Stream[:, k]
    cur_value = a @ theta_star
    high_value = max(high_value, cur_value)


# Set up the parameters for the involved algorithms

N = 30
M = 20
B = 5
T = 300000
L = 1
S = 2.5
delta = 1e-1
Lambda = 2.5
sigma = 0.15

# Define the algorithms


def run_CR_MPS(high_value):

    # the algorithm starts
    regret = np.zeros(1 + T)
    t = 1
    V = Lambda * np.diag(np.ones(d))
    Y = np.zeros(d)
    mu_hat_last = 0
    mu_hat = 0
    a_hat = None
    beta = (np.sqrt(Lambda) * S + sigma * np.sqrt(2 *
            np.log(1 / delta) + d * np.log(1 + T*L*L / (Lambda*d))))**2

    eps_0 = L * S / (3 * np.sqrt(beta))
    left_part = 6 * B * d * np.log(1 + T*L*L / (Lambda*d)) / T

    eps_1 = left_part**(2**(B-1) / (2**(B+1) - 1))
    eps_1 *= eps_0**((2**B - 1) / (2**(B+1) - 1))

    eps_list = np.zeros(B+1)  # eps_list[b] records the value of $\epsilon_b$
    for b in range(1, 1 + B):
        value = eps_1**((2**b-1) / (2**(b-1)))
        value *= eps_0**(- (2**(b-1)-1) / (2**(b-1)))
        eps_list[b] = value
    eps_list[0] = eps_0

    # begin scanning the pass

    for b in range(1, 1 + B):
        #
        a_hat = None
        mu_hat = -1e5

        order = np.random.permutation(K)
        for k in range(K):
            a = Arm_Stream[:, order[k]]
            a_reshape = a.reshape(d, 1)

            theta_hat = np.linalg.inv(V) @ Y
            if b < 1.5 or a @ theta_hat >= mu_hat_last - 2 * np.sqrt(beta) * eps_list[b-1]:
                while a @ np.linalg.inv(V) @ a >= eps_list[b]**2 and t <= T:
                    r_t = theta_star @ a + sigma * np.random.randn()
                    V += a_reshape @ a_reshape.T
                    Y += r_t * a

                    regret[t] = regret[t-1] + high_value - (theta_star @ a)
                    t += 1

                theta_hat = np.linalg.inv(V) @ Y
                if a_hat is None or a @ theta_hat > mu_hat:
                    mu_hat = a @ theta_hat
                    a_hat = a
        mu_hat_last = mu_hat

    while t <= T:
        regret[t] = regret[t-1] + high_value - (theta_star @ a_hat)
        t += 1

    return regret


def run_Agarwal(high_value):
    # Implement the Memory Bounded Successive Elimination algorithm in Agrawal et al., 2022, COLT

    # the algorithm starts
    regret = np.zeros(1 + T)
    Memory = {}
    N = 1.0
    t = 1
    i_tilde = 0
    l_tilde = 0

    for b in range(1, 1 + B):

        N = np.sqrt(N) * T**((2**B) / (2**(B+1)-1))
        for arm in Memory:
            (Memory[arm])[0] = 0.0
            (Memory[arm])[1] = 0.0

        ind = 0
        order = np.random.permutation(K)
        while ind < K and t <= T:

            # read new arms in this pass into the memory
            while ind < K and len(Memory) < M - 1:
                k = order[ind]  # the current arm in the stream is k
                ind += 1
                if Memory.get(k) is None:
                    Memory[k] = np.array([0.0, 0.0])

            # find the least played arm in the memory
            i_min = 0
            n_i_min = 2 * T
            for arm in Memory:
                if (Memory[arm])[0] < n_i_min:
                    i_min = arm
                    n_i_min = (Memory[arm])[0]

            if n_i_min >= N/(K * B):
                # discard an arbitrary arm from the memory
                arm = np.random.choice(list(Memory.keys()))
                del Memory[arm]
            else:
                if t <= T:
                    # play arm i_min once
                    a = Arm_Stream[:, i_min]
                    r_t = a @ theta_star + sigma * np.random.randn()
                    (Memory[i_min])[0] = (Memory[i_min])[0] + 1
                    (Memory[i_min])[1] = (Memory[i_min])[1] + r_t

                    regret[t] = regret[t-1] + high_value - (a @ theta_star)
                    t += 1
            # update l_tilde and i_tilde
            l_tilde = -1e5
            i_tilde = 0
            for arm in Memory:
                if (Memory[arm])[0] < 0.5:
                    continue
                value = ((Memory[arm])[1] / (Memory[arm])[0]) - \
                    sigma * np.sqrt((5 * np.log(T)) / (Memory[arm])[0])
                if value > l_tilde:
                    i_tilde = arm
                    l_tilde = value

            # rule out empirically sub-optimal arms
            elim_list = []
            for arm in Memory:
                if (Memory[arm])[0] < 0.5:
                    continue
                value = ((Memory[arm])[1] / (Memory[arm])[0]) + \
                    sigma * np.sqrt((5 * np.log(T)) / (Memory[arm])[0])
                if value < l_tilde:
                    elim_list.append(arm)
            for arm in elim_list:
                del Memory[arm]

    # play i_tilde until the game terminates

    a_hat = Arm_Stream[:, i_tilde]
    while t <= T:
        regret[t] = regret[t-1] + high_value - (theta_star @ a_hat)
        t += 1

    return regret


def run_Li(high_value):
    # Implement the Multi-Pass Successive Elimination in Li et al., 2023 arXiv

    # the algorithm starts
    regret = np.zeros(1 + T)
    x = -1
    y = -1
    r_y_bar = 0.0
    n_y = 0.0

    t = 1

    for p in range(1, 1 + B):
        #
        order = np.random.permutation(K)
        if p < 1.5:
            y = order[0]
        x = order[0]
        r_x_bar = 0.0
        n_x = 0.0
        beta = (2**(B-p) * (2**p - 1)) / (2**(B+1)-1)
        b = ((T * B)**(2 * beta)) * K**(-2 * beta)

        ind = 1
        while ind < K and t <= T:
            while n_x <= b or n_y <= b:
                #
                flip = np.random.rand()
                tag = "y"
                a = None
                if n_x < n_y or (abs(n_x-n_y) < 0.01 and flip < 0.5):
                    tag = "x"
                if tag == "y":
                    a = Arm_Stream[:, y]
                    r_t = a @ theta_star + sigma * np.random.randn()
                    r_y_bar = (r_y_bar * n_y + r_t) / (n_y + 1)
                    n_y += 1
                else:
                    a = Arm_Stream[:, x]
                    r_t = a @ theta_star + sigma * np.random.randn()
                    r_x_bar = (r_x_bar * n_x + r_t) / (n_x + 1)
                    n_x += 1

                if t <= T:
                    regret[t] = regret[t-1] + high_value - (theta_star @ a)
                    t += 1
                else:
                    break

                if n_x > 0.5 and n_y > 0.5 and r_x_bar + sigma * np.sqrt((5 * np.log(T))/n_x) < r_y_bar - sigma * np.sqrt((5 * np.log(T))/n_y):
                    break
                elif n_x > 0.5 and n_y > 0.5 and r_y_bar + sigma * np.sqrt((5 * np.log(T))/n_y) < r_x_bar - sigma * np.sqrt((5 * np.log(T))/n_x):
                    y = x
                    r_y_bar = r_x_bar
                    n_y = n_x
                    break

            # read a new x from the arm stream
            x = order[ind]
            ind += 1
            r_x_bar = 0.0
            n_x = 0.0

    # play i_tilde until the game terminates

    a_hat = Arm_Stream[:, y]
    while t <= T:
        regret[t] = regret[t-1] + high_value - (theta_star @ a_hat)
        t += 1

    return regret


# Parameters for the figure
rows = 1
cols = 1
line_width = 1.8

fig, axs = plt.subplots(rows, cols)

regret_CR_MPS = np.zeros(1 + T)
regret_Ag = np.zeros(1 + T)
regret_Li = np.zeros(1 + T)

for i in range(N):
    print("ite=%d/%d" % (i+1, N))
    print("  CR-MPS")
    cur_regret = run_CR_MPS(high_value)
    regret_CR_MPS += cur_regret

    print("  Agarwal")
    cur_regret = run_Agarwal(high_value)
    regret_Ag += cur_regret

    print("  Li")
    cur_regret = run_Li(high_value)
    regret_Li += cur_regret


regret_CR_MPS /= N
regret_Ag /= N
regret_Li /= N

axs.plot(np.arange(1, 1 + T),
         regret_CR_MPS[1:], linewidth=line_width, label=r"CR-MPS (Our Approach)")
axs.plot(np.arange(1, 1 + T),
         regret_Ag[1:], linewidth=line_width, label=r"MBSE (Agarwal et al., 2022)")
axs.plot(np.arange(1, 1 + T),
         regret_Li[1:], linewidth=line_width, label=r"MPSE (Li et al., 2023)")

axs.set_xlabel(r"Time horizon $T$")
axs.set_ylabel(r"Cumulative Regret")
axs.set_xlim(0, 15000)
axs.set_ylim(0, 17000)
axs.set_title(r"Dataset: Employee’s Performance for HR Analytics")
axs.grid(1)
axs.legend()

plt.show()
