import numpy as np
import control

from scipy import signal
# import random

# The number of robots, target positions, initial positions
num_robot = 14

target_x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -3.0, 3.0])
target_y = np.array([2.0, 2.0, 1.0, 2.0, 2.0, 0.0, -1.0, -2.0, -3.0, -2.0, -1.0, 0.0, 1.0, 1.0])
z_target = target_x + 1j * target_y

pos_x = target_x + 3 * (np.random.rand(num_robot) - 0.5)
pos_y = target_y + 3 * (np.random.rand(num_robot) - 0.5)
z = pos_x + 1j * pos_y

# Define connectivity
A = np.zeros((num_robot, num_robot), dtype=bool)
A[0, (1, 12)] = True
A[1, (0, 2)] = True
A[2, (1, 3)] = True
A[3, (2, 4)] = True
A[4, (3, 13)] = True
A[5, (6, 13)] = True
A[6, (5, 7)] = True
A[7, (6, 8)] = True
A[8, (7, 9)] = True
A[9, (8, 10)] = True
A[10, (9, 11)] = True
A[11, (10, 12)] = True
A[12, (0, 11)] = True
A[13, (4, 5)] = True

num_edge = int(len(np.argwhere(A == True)) / 2)

print(num_edge)

# Simulation
dt = 0.1
alpha = 0.05
# poles = np.ones(num_robot - 2, dtype=complex)
poles_list = [1.0 + 1j * 1.0 for i in range(num_robot - 2)]
num_iteration = 200

for iteration in range(num_iteration):
    # Determine edge weights
    L = np.zeros((num_robot, num_robot), dtype=complex)
    for i in range(num_robot):
        for j in range(num_robot - 1):
            if j == i:
                continue
            for k in range(j + 1, num_robot):
                if k == i or (A[i, j] == False) or (A[i, k] == False):
                    continue
                L[i, j] = L[i, j] + z_target[k] - z_target[i]
                L[i, k] = L[i, k] + z_target[i] - z_target[j]
        L[i, i] = -np.sum(L[i, :])
    # Design K
    U_svd, s_svd, V_svd = np.linalg.svd(L)
    s_sqrt = np.sqrt(s_svd)
    S_sqrt = np.diag(s_sqrt)
    U = U_svd[:, :-2] @ S_sqrt[:-2, :-2]
    V = S_sqrt[:-2, :-2] @ V_svd[:-2, :]
    pp_arg1 = np.zeros((num_robot - 2, num_robot - 2), dtype=complex)
    pp_arg2 = U[:-2, :] @ V[:, :-2]
    print(pp_arg1)
    print(pp_arg2)
    # temp = control.acker(pp_arg1, pp_arg2, poles_list)
    # temp = signal.place_poles(pp_arg1, pp_arg2, poles_list)
    # print(temp)
    # K = np.concatenate
