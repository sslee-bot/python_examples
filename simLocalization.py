import numpy as np
import sympy as sym
from scipy.io import loadmat
import estimator
import matplotlib.pyplot as plt

# State variables
x, y, theta = sym.symbols('x y theta')
# Control inputs, Time-varying parameters
v_l, v_theta, dt = sym.symbols('v_l v_theta dt')
state_set = (x, y, theta)
input_set = (v_l, v_theta, dt)
# Position values of fixed anchors #1 ~ #4
anchor_pos = np.array([[-3.0, -3.0], [-3.0, 3.0], [3.0, 3.0], [3.0, -3.0]])
# State and measurement equations
f = sym.Matrix(1, 3, [0, 0, 0])
f[0] = x + v_l * dt * sym.cos(theta + 0.5 * v_theta * dt)
f[1] = y + v_l * dt * sym.sin(theta + 0.5 * v_theta * dt)
f[2] = theta + v_theta * dt
h = sym.Matrix(1, 5, [0, 0, 0, 0, 0])
h[0] = sym.sqrt((x - anchor_pos[0][0]) ** 2 + (y - anchor_pos[0][1]) ** 2)
h[1] = sym.sqrt((x - anchor_pos[1][0]) ** 2 + (y - anchor_pos[1][1]) ** 2)
h[2] = sym.sqrt((x - anchor_pos[2][0]) ** 2 + (y - anchor_pos[2][1]) ** 2)
h[3] = sym.sqrt((x - anchor_pos[3][0]) ** 2 + (y - anchor_pos[3][1]) ** 2)
h[4] = theta

# Load measurement, control input data
RTLS_data = loadmat('Exp_data.mat')
msr_data = RTLS_data['measurement_data']
ctrl_data = RTLS_data['control_data']
sampling_time = RTLS_data['sampling_time'][0]

# Initialize estimator
x_init = np.array([2.5, 0.5, 0])
z_init = msr_data[0]
Est1 = estimator.EKF(f, h, state_set, input_set)
Est2 = estimator.FIR(f, h, state_set, input_set, 15, x_init, z_init)

# Simulation
num_iteration = msr_data.shape[0]
EKF_data = np.zeros((num_iteration, len(state_set)))
FIR_data = np.zeros((num_iteration, len(state_set)))
for i in range(num_iteration):
    msr = msr_data[i]
    ctrl = np.concatenate((ctrl_data[i], np.array(sampling_time)))
    alpha = not (0 in msr)

    x_hat_EKF = Est1.estimate(msr, ctrl, alpha)
    x_hat_FIR = Est2.estimate(msr, ctrl, alpha)
    EKF_data[i] = x_hat_EKF
    FIR_data[i] = x_hat_FIR

# Plot Results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(EKF_data[:, 0], EKF_data[:, 1])
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('EKF')
plt.subplot(122)
plt.plot(FIR_data[:, 0], FIR_data[:, 1])
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('FIR')
plt.show()
