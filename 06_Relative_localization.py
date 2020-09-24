import numpy as np
import sympy as sym
from modules import estimator
import matplotlib.pyplot as plt

# State variables, Control inputs, Time-varying parameters
x1, y1, theta1 = sym.symbols('x1 y1 theta1')
x2, y2, theta2 = sym.symbols('x2 y2 theta2')
v_l1, v_theta1 = sym.symbols('v_l1 v_theta1')
v_l2, v_theta2 = sym.symbols('v_l2 v_theta2')
dt = sym.symbols('dt')
state_set = (x1, y1, theta1, x2, y2, theta2)
input_set = (v_l1, v_theta1, v_l2, v_theta2, dt)
# State and measurement equations
f = sym.Matrix(1, 6, [0, 0, 0, 0, 0, 0])
f[0] = x1 + v_l1 * dt * sym.cos(theta1 + 0.5 * v_theta1 * dt)
f[1] = y1 + v_l1 * dt * sym.sin(theta1 + 0.5 * v_theta1 * dt)
f[2] = theta1 + v_theta1 * dt
f[3] = x2 + v_l2 * dt * sym.cos(theta2 + 0.5 * v_theta2 * dt)
f[4] = y2 + v_l2 * dt * sym.sin(theta2 + 0.5 * v_theta2 * dt)
f[5] = theta2 + v_theta2 * dt
h = sym.Matrix(1, 5, [0, 0, 0, 0, 0])
h[0] = sym.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # distance
h[1] = x1
h[2] = y1
h[3] = theta1
h[4] = theta2  # own heading angle

# Generate real values
num_iteration = 50
sampling_time = 0.5
ctrl_data1 = np.zeros((num_iteration, 2))
ctrl_data2 = np.zeros((num_iteration, 2))
msr_data = np.zeros((num_iteration, 5))
state_data = np.zeros((num_iteration, 6))
temp = np.zeros(6)
x1_init = np.array([-1.0, 1.0, 0.7])
x2_init = np.array([1.0, -1.0, -0.5])
x1_real, y1_real, theta1_real = x1_init[0], x1_init[1], x1_init[2]
x2_real, y2_real, theta2_real = x2_init[0], x2_init[1], x2_init[2]

for i in range(num_iteration):
    ctrl_data1[i, 0], ctrl_data1[i, 1] = 0.2, -0.1
    ctrl_data2[i, 0], ctrl_data2[i, 1] = 0.2, 0.1
    msr_data[i] = np.array(h.subs([(x1, x1_real), (y1, y1_real), (theta1, theta1_real), (x2, x2_real), (y2, y2_real),
                                   (theta2, theta2_real)])).astype(np.float)
    temp = np.array(f.subs(
        [(x1, x1_real), (y1, y1_real), (theta1, theta1_real), (x2, x2_real), (y2, y2_real), (theta2, theta2_real),
         (v_l1, ctrl_data1[i, 0]), (v_theta1, ctrl_data1[i, 1]), (v_l2, ctrl_data2[i, 0]), (v_theta2, ctrl_data2[i, 1]),
         (dt, sampling_time)])).astype(np.float)
    x1_real, y1_real, theta1_real = temp[0][0], temp[0][1], temp[0][2]
    x2_real, y2_real, theta2_real = temp[0][3], temp[0][4], temp[0][5]
    state_data[i] = np.array([x1_real, y1_real, theta1_real, x2_real, y2_real, theta2_real])

# Initialize estimator
x_init = np.concatenate((x1_init, x2_init)) + [0.0, 0.0, 0.0, 0.3, 0.3, 0.1]    # Suppose initial error
z_init = msr_data[0]
u_init = np.concatenate((ctrl_data1[0], ctrl_data2[0], [sampling_time]), axis=0)

K = np.kron(np.diag([1.0, 3.0, 5.0, 7.0, 9.0]), np.diag([0.1, 1.0, 1.0, 1.0, 1.0]))
Est = estimator.FIR_partial(f, h, state_set, input_set, 5, 3.0, K, x_init, z_init, u_init)

# Simulation
FIR_partial_data = np.zeros((num_iteration, len(state_set)))
for i in range(num_iteration):
    print(i)
    msr = msr_data[i]
    ctrl = np.concatenate((ctrl_data1[i], ctrl_data2[i], [sampling_time]))

    x_hat_FIR_partial = Est.estimate(msr, ctrl)
    FIR_partial_data[i] = x_hat_FIR_partial

figure_2d = plt.figure(1)
plt.plot(state_data[:, 0], state_data[:, 1], 'o', label='Robot 1')
plt.plot(state_data[:, 3], state_data[:, 4], 'o', label='Robot 2 (real)')
plt.plot(FIR_partial_data[:, 3], FIR_partial_data[:, 4], '*', label='Robot 2 (estimated)')
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid()
plt.legend()

figure_error = plt.figure(2)
plt.subplot(211)
plt.plot(range(num_iteration), state_data[:, 3] - FIR_partial_data[:, 3])
plt.grid()
plt.ylabel('X error (m)')
plt.subplot(212)
plt.plot(range(num_iteration), state_data[:, 4] - FIR_partial_data[:, 4])
plt.grid()
plt.ylabel('Y error (m)')
plt.xlabel('Iteration')
plt.show()
