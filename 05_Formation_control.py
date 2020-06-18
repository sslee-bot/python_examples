import numpy as np
import matplotlib.pyplot as plt

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

# Simulation
dt = 0.1
alpha = 0.05
# poles = np.ones(num_robot - 2)
# poles = np.ones(num_robot - 2, dtype=complex)

# poles_list = [1.0 + 1j * 1.0 for i in range(num_robot - 2)]
poles_list = [0.1 + 1j * 0.0 for i in range(num_robot - 2)]
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
    K = np.linalg.inv(V[:, :-2]) @ -np.diag(poles_list) @ np.linalg.inv(U[:-2, :])
    # Control
    z_dot = np.zeros_like(z)
    for i in range(num_robot - 2):
        for j in range(num_robot):
            if j == i:
                continue
            z_dot[i] = z_dot[i] + K[i][i] * L[i][j] * (z[j] - z[i])
    d_bar = np.linalg.norm(z_target[-1] - z_target[-2])
    diff_leaders = z[-1] - z[-2]
    z_dot[-2] = alpha * diff_leaders * (np.linalg.norm(diff_leaders) ** 2 - d_bar ** 2)
    z_dot[-1] = -z_dot[-2]
    z = z + z_dot * dt

# Result
plt.plot(z_target.real, z_target.imag, '*', label='References')
plt.plot(z.real, z.imag, 'o', label='Robots')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.show()
